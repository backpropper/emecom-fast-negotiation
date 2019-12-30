import torch


def calc_rewards(t, s, term, device):
    # calcualate rewards for any that just finished
    # it will calculate three reward values:
    # agent 1 (as proporition of max agent 1), agent 2 (as proportion of max agent 2), prosocial (as proportion of max prosocial)
    # in the non-prosocial setting, we need all three:
    # - first two for training
    # - next one for evaluating Table 1, in the paper
    # in the prosocial case, we'll skip calculating the individual agent rewards, possibly/probably

    # assert prosocial, 'not tested for not prosocial currently'

    agent = t % 2
    batch_size = term.size()[0]
    rewards_batch = torch.zeros(batch_size, 3, dtype=torch.float, device=device)  # each row is: {one, two, combined}
    if t == 0:
        return rewards_batch

    reward_eligible_mask = term.view(batch_size).clone().byte()
    if reward_eligible_mask.max() == 0:
        return rewards_batch

    exceeded_pool, _ = ((s.last_proposal - s.pool) > 0).max(1)
    if exceeded_pool.max() > 0:
        reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0
        if reward_eligible_mask.max() == 0:
            return rewards_batch

    proposer = 1 - agent
    accepter = agent
    proposal = torch.zeros(batch_size, 2, 3, dtype=torch.long, device=device)
    proposal[:, proposer] = s.last_proposal
    proposal[:, accepter] = s.pool - s.last_proposal
    max_utility, _ = s.utilities.max(1)

    reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
    for b in reward_eligible_idxes:
        raw_rewards = torch.zeros(2, dtype=torch.float, device=device)
        for i in range(2):
            raw_rewards[i] = s.utilities[b, i].float().dot(proposal[b, i].float())

        scaled_rewards = torch.zeros(3, dtype=torch.float, device=device)

        actual_prosocial = raw_rewards.sum()
        available_prosocial = max_utility[b].float().dot(s.pool[b].float())
        if available_prosocial != 0:
            scaled_rewards[2] = actual_prosocial / available_prosocial

        for i in range(2):
            max_agent = s.utilities[b, i].float().dot(s.pool[b].float())
            if max_agent != 0:
                scaled_rewards[i] = raw_rewards[i] / max_agent

        rewards_batch[b] = scaled_rewards
    return rewards_batch
