import json
import time
import argparse
import os
import sys
import datetime
import numpy as np
import random

import torch
from torch import optim

import nets
import sampling
import rewards_lib
import alive_sieve


def render_action(t, s, prop, term):
    agent = t % 2
    speaker = 'A' if agent == 0 else 'B'
    utility = s.utilities[:, agent]
    print('  ', end='')
    if speaker == 'B':
        print('                                   ', end='')
    if term[0][0]:
        print(' ACC')
    else:
        print(' ' + ''.join([str(v) for v in s.m_prev[0].view(-1).tolist()]), end='')
        print(' %s:%s/%s %s:%s/%s %s:%s/%s' % (
            int(utility[0][0]), int(prop[0][0]), int(s.pool[0][0]),
            int(utility[0][1]), int(prop[0][1]), int(s.pool[0][1]),
            int(utility[0][2]), int(prop[0][2]), int(s.pool[0][2]),
        ), end='')
        print('')
        if t + 1 == s.N[0]:
            print('  [out of time]')


def save_model(model_file, agent_models, agent_opts, start_time, episode):
    state = {}
    for i in range(2):
        state['agent%s' % i] = {}
        state['agent%s' % i]['model_state'] = agent_models[i].state_dict()
        state['agent%s' % i]['opt_state'] = agent_opts[i].state_dict()
    state['episode'] = episode
    state['elapsed_time'] = time.time() - start_time
    with open(model_file + '.tmp', 'wb') as f:
        torch.save(state, f)
    os.rename(model_file + '.tmp', model_file)


def load_model(model_file, agent_models, agent_opts):
    with open(model_file, 'rb') as f:
        state = torch.load(f)
    for i in range(2):
        agent_models[i].load_state_dict(state['agent%s' % i]['model_state'])
        agent_opts[i].load_state_dict(state['agent%s' % i]['opt_state'])
    episode = state['episode']
    # create a kind of 'virtual' start_time
    start_time = time.time() - state['elapsed_time']
    return episode, start_time


class State(object):
    def __init__(self, N, pool, utilities, device):
        batch_size = N.size()[0]
        self.N = N.to(device=device)
        self.pool = pool.to(device=device)
        self.utilities = torch.zeros(batch_size, 2, 3, dtype=torch.long, device=device)
        self.utilities[:, 0] = utilities[0]
        self.utilities[:, 1] = utilities[1]

        self.last_proposal = torch.zeros(batch_size, 3, dtype=torch.long, device=device)
        self.m_prev = torch.zeros(batch_size, 6, dtype=torch.long, device=device)

    def sieve_(self, still_alive_idxes):
        self.N = self.N[still_alive_idxes]
        self.pool = self.pool[still_alive_idxes]
        self.utilities = self.utilities[still_alive_idxes]
        self.last_proposal = self.last_proposal[still_alive_idxes]
        self.m_prev = self.m_prev[still_alive_idxes]


def run_episode(batch, device, enable_comms, enable_proposal, prosocial, agent_models, testing, corrupt_utt, render=False):
    batch_size = batch['N'].size()[0]
    s = State(**batch, device=device)

    sieve = alive_sieve.AliveSieve(batch_size=batch_size, device=device)
    actions_by_timestep = []
    alive_masks = []

    rewards = torch.zeros((batch_size, 3), device=device, dtype=torch.float)
    num_steps = torch.ones(batch_size, device=device, dtype=torch.long) * 10

    term_matches_argmax_count = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    num_policy_runs = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0

    entropy_loss_by_agent = [0, 0]
    if render:
        print('  ')
    for t in range(10):
        agent = t % 2

        agent_model = agent_models[agent]
        if enable_comms:
            _prev_message = s.m_prev
        else:
            _prev_message = torch.zeros((sieve.batch_size, 6), dtype=torch.long, device=device)
        if enable_proposal:
            _prev_proposal = s.last_proposal
        else:
            _prev_proposal = torch.zeros((sieve.batch_size, 3), dtype=torch.long, device=device)
        nodes, term_a, s.m_prev, this_proposal, _entropy_loss, \
                _term_matches_argmax_count, _utt_matches_argmax_count, _utt_stochastic_draws, \
                _prop_matches_argmax_count, _prop_stochastic_draws = agent_model(pool=s.pool,
                                                                                utility=s.utilities[:, agent],
                                                                                m_prev=s.m_prev,
                                                                                prev_proposal=_prev_proposal,
                                                                                testing=testing,
                                                                                corrupt_utt=corrupt_utt
                                                                                )
        entropy_loss_by_agent[agent] += _entropy_loss
        actions_by_timestep.append(nodes)
        term_matches_argmax_count += _term_matches_argmax_count
        num_policy_runs += sieve.batch_size
        utt_matches_argmax_count += _utt_matches_argmax_count
        utt_stochastic_draws += _utt_stochastic_draws
        prop_matches_argmax_count += _prop_matches_argmax_count
        prop_stochastic_draws += _prop_stochastic_draws

        if render and sieve.out_idxes[0] == 0:
            render_action(t=t, s=s, term=term_a, prop=this_proposal)

        new_rewards = rewards_lib.calc_rewards(t=t, s=s, term=term_a, device=device)
        rewards[sieve.out_idxes] = new_rewards
        s.last_proposal = this_proposal

        sieve.mark_dead(term_a)
        sieve.mark_dead(t + 1 == s.N)
        alive_masks.append(sieve.alive_mask.clone())
        sieve.set_dead_global(num_steps, t + 1)
        if sieve.all_dead():
            break

        s.sieve_(sieve.alive_idxes)
        sieve.self_sieve_()

    if render:
        print('  r: %.2f' % rewards[0][2])
        print('  ')

    return actions_by_timestep, rewards, num_steps, alive_masks, entropy_loss_by_agent, \
        term_matches_argmax_count, num_policy_runs, utt_matches_argmax_count, utt_stochastic_draws, \
        prop_matches_argmax_count, prop_stochastic_draws


def safe_div(a, b):
    """
    returns a / b, unless b is zero, in which case returns 0

    this is primarily for usage in cases where b might be systemtically zero, eg because comms are disabled or similar
    """
    return 0 if b == 0 else a / b


def run(enable_proposal, enable_comms, seed, prosocial, logfile, model_file, batch_size,
        term_entropy_reg, utterance_entropy_reg, proposal_entropy_reg, device,
        no_load, testing, test_seed, render_every_seconds):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_r = np.random.RandomState(seed)
    else:
        train_r = np.random

    test_r = np.random.RandomState(test_seed)
    test_batches = sampling.generate_test_batches(batch_size=batch_size, num_batches=5, random_state=test_r)
    test_hashes = sampling.hash_batches(test_batches)

    episode = 0
    start_time = time.time()
    agent_models = []
    agent_opts = []
    for i in range(2):
        model = nets.AgentModel(enable_comms=enable_comms,
                            enable_proposal=enable_proposal,
                            device=device,
                            term_entropy_reg=term_entropy_reg,
                            utterance_entropy_reg=utterance_entropy_reg,
                            proposal_entropy_reg=proposal_entropy_reg
                            )
        model = model.to(device=device)
        agent_models.append(model)
        agent_opts.append(optim.Adam(params=agent_models[i].parameters()))

    if os.path.isfile(model_file) and not no_load:
        episode, start_time = load_model(
            model_file=model_file,
            agent_models=agent_models,
            agent_opts=agent_opts)
        print('loaded model')
    elif testing:
        print('')
        print('ERROR: must have loadable model to use --testing option')
        print('')
        return

    for d in ['logs', 'model_saves']:
        if not os.path.isdir(d):
            os.makedirs(d)
    f_log = open(logfile, 'w')
    json_dict = {'enable_proposal': enable_proposal,
                'enable_comms': enable_comms,
                'prosocial': prosocial,
                'seed': seed
                }
    f_log.write('meta: %s\n' % json.dumps(json_dict))

    last_print = time.time()
    last_save = time.time()
    steps_sum = 0
    count_sum = 0
    rewards_sum = torch.zeros(3, dtype=torch.float, device=device)
    baseline = torch.zeros(3, dtype=torch.float, device=device)
    term_matches_argmax_count = 0
    num_policy_runs = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0

    while True:
        render = time.time() - last_print >= render_every_seconds
        batch = sampling.generate_training_batch(batch_size=batch_size, test_hashes=test_hashes, random_state=train_r)
        p = random.uniform(0, 1)
        if p >= 0.5:
            corrupt_utt = True
        else:
            corrupt_utt = False
        actions, rewards, steps, alive_masks, entropy_loss_by_agent, \
                _term_matches_argmax_count, _num_policy_runs, _utt_matches_argmax_count, _utt_stochastic_draws, \
                _prop_matches_argmax_count, _prop_stochastic_draws = run_episode(batch=batch,
                                                                                device=device,
                                                                                enable_comms=enable_comms,
                                                                                enable_proposal=enable_proposal,
                                                                                agent_models=agent_models,
                                                                                prosocial=prosocial,
                                                                                render=render,
                                                                                testing=testing,
                                                                                corrupt_utt=corrupt_utt
                                                                                )
        term_matches_argmax_count += float(_term_matches_argmax_count)
        utt_matches_argmax_count += float(_utt_matches_argmax_count)
        utt_stochastic_draws += float(_utt_stochastic_draws)
        num_policy_runs += float(_num_policy_runs)
        prop_matches_argmax_count += float(_prop_matches_argmax_count)
        prop_stochastic_draws += float(_prop_stochastic_draws)

        if not testing:
            for i in range(2):
                agent_opts[i].zero_grad()
            reward_loss_by_agent = [0, 0]
            baselined_rewards = rewards - baseline
            rewards_by_agent = []
            for i in range(2):
                if prosocial:
                    rewards_by_agent.append(baselined_rewards[:, 2])
                else:
                    rewards_by_agent.append(baselined_rewards[:, i])
            sieve_playback = alive_sieve.SievePlayback(alive_masks, device=device)
            for t, global_idxes in sieve_playback:
                agent = t % 2
                if len(actions[t]) > 0:
                    for action in actions[t]:
                        _rewards = rewards_by_agent[agent]
                        _reward = _rewards[global_idxes].float().contiguous().view(sieve_playback.batch_size, 1)
                        _reward_loss = -(action * _reward)
                        _reward_loss = _reward_loss.sum()
                        reward_loss_by_agent[agent] += _reward_loss
            for i in range(2):
                loss = entropy_loss_by_agent[i] + reward_loss_by_agent[i]
                loss.backward()
                agent_opts[i].step()

        rewards_sum += rewards.sum(0)
        steps_sum += float(steps.sum())
        baseline = 0.7 * baseline + 0.3 * rewards.mean(0)
        count_sum += batch_size

        if render:
            test_rewards_sum = 0
            for test_batch in test_batches:
                actions, test_rewards, steps, alive_masks, entropy_loss_by_agent, \
                        _term_matches_argmax_count, _num_policy_runs, _utt_matches_argmax_count, _utt_stochastic_draws, \
                        _prop_matches_argmax_count, _prop_stochastic_draws = run_episode(batch=test_batch,
                                                                                        device=device,
                                                                                        enable_comms=enable_comms,
                                                                                        enable_proposal=enable_proposal,
                                                                                        agent_models=agent_models,
                                                                                        prosocial=prosocial,
                                                                                        render=True,
                                                                                        testing=True,
                                                                                        corrupt_utt=False
                                                                                        )
                test_rewards_sum += test_rewards[:, 2].mean()

            test_rewards_sum = float(test_rewards_sum)
            rewards_sum_pr = rewards_sum.cpu().float().numpy()
            baseline_pr = baseline.cpu().float().numpy()
            print('test reward=%.3f' % (test_rewards_sum / len(test_batches)))

            time_since_last = time.time() - last_print
            if prosocial:
                baseline_str = '%.2f' % baseline[2]
                rewards_str = '%.2f' % (rewards_sum_pr[2] / count_sum)
            else:
                baseline_str = '%.2f,%.2f' % (baseline_pr[0], baseline_pr[1])
                rewards_str = '%.2f,%.2f' % (rewards_sum_pr[0] / count_sum, rewards_sum_pr[1] / count_sum)
            print('e=%s train=%s b=%s games/sec %s avg_steps %.4f argmaxp term=%.4f utt=%.4f prop=%.4f' % (
                episode,
                rewards_str,
                baseline_str,
                int(count_sum / time_since_last),
                steps_sum / count_sum,
                term_matches_argmax_count / num_policy_runs,
                safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                prop_matches_argmax_count / prop_stochastic_draws
            ))

            f_log.write(json.dumps({
                'episode': episode,
                'avg_reward_pro': rewards_sum_pr[2] / count_sum,
                'test_reward': test_rewards_sum / len(test_batches),
                'avg_steps': steps_sum / count_sum,
                'games_sec': count_sum / time_since_last,
                'elapsed': time.time() - start_time,
                'argmaxp_term': (term_matches_argmax_count / num_policy_runs),
                'argmaxp_utt': safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                'argmaxp_prop': (prop_matches_argmax_count / prop_stochastic_draws)
            }) + '\n')
            f_log.flush()

            last_print = time.time()
            steps_sum = 0
            count_sum = 0
            rewards_sum = torch.zeros(3, dtype=torch.float, device=device)
            term_matches_argmax_count = 0
            num_policy_runs = 0
            utt_matches_argmax_count = 0
            utt_stochastic_draws = 0
            prop_matches_argmax_count = 0
            prop_stochastic_draws = 0

        if not testing and time.time() - last_save >= render_every_seconds:
            save_model(
                model_file=model_file,
                agent_models=agent_models,
                agent_opts=agent_opts,
                start_time=start_time,
                episode=episode)
            print('saved model')
            last_save = time.time()

        episode += 1
    f_log.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='model_saves/model.pt')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-seed', type=int, default=123, help='used for generating test game set')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--term-entropy-reg', type=float, default=0.5)
    parser.add_argument('--utterance-entropy-reg', type=float, default=0.0001)
    parser.add_argument('--proposal-entropy-reg', type=float, default=0.01)
    parser.add_argument('--disable-proposal', action='store_true')
    parser.add_argument('--disable-comms', action='store_true')
    parser.add_argument('--disable-prosocial', action='store_true')
    parser.add_argument('--render-every-seconds', type=int, default=30)
    parser.add_argument('--testing', action='store_true', help='turn off learning; always pick argmax')
    parser.add_argument('--no-load', action='store_true')
    parser.add_argument('--name', type=str, default='', help='used for logfile naming')
    parser.add_argument('--logfile', type=str, default='logs/log_%Y%m%d_%H%M%S{name}.log')
    args = parser.parse_args()

    args.enable_comms = not args.disable_comms
    args.enable_proposal = not args.disable_proposal
    args.prosocial = not args.disable_prosocial
    args.logfile = args.logfile.format(**args.__dict__)
    args.logfile = datetime.datetime.strftime(datetime.datetime.now(), args.logfile)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    del args.__dict__['disable_comms']
    del args.__dict__['disable_proposal']
    del args.__dict__['disable_prosocial']
    del args.__dict__['name']

    run(**args.__dict__)
