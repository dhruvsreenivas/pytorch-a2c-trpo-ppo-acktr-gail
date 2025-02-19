import torch
import torch.nn as nn
import torch.nn.functional as F
from low_precision_utils import *


def conjugate_gradients(Avp, b, n_steps, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(n_steps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def backtracking_line_search(model, f, full_step, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    prev_parameters = model.flat_actor_parameters()
    with torch.no_grad():
        initial_f_value = f()
    for step_frac in .5**torch.arange(max_backtracks):
        new_parameters = prev_parameters + step_frac * full_step
        model.set_flat_actor_parameters(new_parameters)
        with torch.no_grad():
            new_f_value = f()
        actual_improve = initial_f_value - new_f_value
        expected_improve = expected_improve_rate * step_frac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            model.set_flat_actor_parameters(new_parameters)
            return
    model.set_flat_actor_parameters(prev_parameters)


class TRPO:
    def __init__(self,
                 actor_critic: nn.Module,
                 batch_size,
                 lr,
                 eps,
                 l2_reg=1e-2,
                 max_kl=1e-4,
                 damping=1e-3,
                 use_hadam=False):
        self.actor_critic = actor_critic

        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.max_kl = max_kl
        self.damping = damping

        if use_hadam:
            self.optimizer = hAdam(
                actor_critic.base.critic.parameters(), lr=lr, eps=eps, weight_decay=l2_reg)
        self.optimizer = torch.optim.Adam(
            actor_critic.base.critic.parameters(), lr=lr, eps=eps, weight_decay=l2_reg)

    def update(self, rollouts):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)

        value_loss = 0
        action_loss = 0
        dist_entropy = 0

        if self.actor_critic.is_recurrent:
            data_generator = rollouts.recurrent_generator(
                advantages, 1, self.batch_size)
        else:
            data_generator = rollouts.feed_forward_generator(
                advantages, 1, self.batch_size)

        for sample in data_generator:
            obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            self.actor_critic.actor_requires_grad(False)
            self.actor_critic.critic_requires_grad(True)
            values = self.actor_critic.get_value(
                obs_batch, recurrent_hidden_states_batch, masks_batch)

            # Value Loss calculation and value network optimization:====================================================
            self.optimizer.zero_grad()
            value_loss = F.mse_loss(return_batch, values)
            value_loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                _, current_action_log_probs, dist_entropy, _ = \
                    self.actor_critic.evaluate_actions(obs_batch, recurrent_hidden_states_batch,
                                                       masks_batch, actions_batch)
            self.actor_critic.actor_requires_grad(True)
            self.actor_critic.critic_requires_grad(False)

            # Policy Loss calculation and TRPO step:====================================================================
            def get_policy_loss():
                _, action_log_probs, _, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                action_loss = - adv_targ * \
                    torch.exp(action_log_probs - old_action_log_probs_batch)
                return action_loss.mean()

            prev_policy = self.actor_critic.dist_layer.dist

            def get_kl():
                current_policy = self.actor_critic.dist_layer.dist
                return torch.distributions.kl_divergence(current_policy, prev_policy)

            current_action_loss = get_policy_loss()
            action_loss_grad = torch.autograd.grad(current_action_loss,
                                                   self.actor_critic.actor_parameters(),
                                                   retain_graph=True)
            action_loss_grad = torch.cat(
                [grad.view(-1) for grad in action_loss_grad]).detach()

            def sample_mean_kl_div_hessian(v):
                kl = get_kl().mean()

                grads = torch.autograd.grad(kl,
                                            self.actor_critic.actor_parameters(),
                                            create_graph=True,
                                            retain_graph=True)
                flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

                kl_v = (flat_grad_kl * v).sum()
                grads = torch.autograd.grad(
                    kl_v, self.actor_critic.actor_parameters(), retain_graph=True)
                flat_grad_grad_kl = torch.cat(
                    [grad.contiguous().view(-1) for grad in grads]).detach()

                return flat_grad_grad_kl + v * self.damping

            step_dir = conjugate_gradients(
                sample_mean_kl_div_hessian, -action_loss_grad, 10)

            shs = 0.5 * \
                (step_dir * sample_mean_kl_div_hessian(step_dir)).sum(0, keepdim=True)

            lm = torch.sqrt(shs / self.max_kl)
            full_step = step_dir / lm[0]

            neg_dot_step_dir = (-action_loss_grad *
                                step_dir).sum(0, keepdim=True)

            backtracking_line_search(
                self.actor_critic, get_policy_loss, full_step, neg_dot_step_dir / lm[0])

            action_loss += current_action_loss

            value_loss += value_loss.item()
            dist_entropy += dist_entropy.item()

        value_loss /= self.batch_size
        action_loss /= self.batch_size
        dist_entropy /= self.batch_size

        return value_loss, action_loss, dist_entropy
