import torch
from torch import nn
import torch.nn.functional as F


class NumberSequenceEncoder(nn.Module):
    def __init__(self, num_values, device, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, embedding_size)
        self.lstm = nn.LSTMCell(input_size=embedding_size, hidden_size=embedding_size)
        self.device = device

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.transpose(0, 1)
        x = self.embedding(x)
        h = torch.zeros(batch_size, self.embedding_size, dtype=torch.float, device=self.device)
        c = torch.zeros(batch_size, self.embedding_size, dtype=torch.float, device=self.device)

        for s in range(seq_len):
            h, c = self.lstm(x[s], (h, c))
        return h


class CombinedNet(nn.Module):
    def __init__(self, num_sources=3, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.h1 = nn.Linear(embedding_size * num_sources, embedding_size)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        return x


class TermPolicy(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.h1 = nn.Linear(embedding_size, 1)

    def forward(self, thoughtvector, testing, eps=1e-8):
        logits = self.h1(thoughtvector)
        term_probs = torch.sigmoid(logits)

        res_greedy = (term_probs.detach() >= 0.5).view(-1, 1).float()

        log_g = None
        if not testing:
            a = torch.bernoulli(term_probs)
            g = a.detach() * term_probs + (1 - a.detach()) * (1 - term_probs)
            log_g = g.log()
            a = a.detach()
        else:
            a = res_greedy

        matches_greedy = res_greedy == a
        matches_greedy_count = matches_greedy.int().sum()
        term_probs = term_probs + eps
        entropy = -(term_probs * term_probs.log()).sum(1).sum()
        return log_g, a.byte(), entropy, matches_greedy_count


class UtterancePolicy(nn.Module):
    def __init__(self, device, embedding_size=100, num_tokens=10, max_len=6):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.lstm = nn.LSTMCell(input_size=embedding_size, hidden_size=embedding_size)
        self.h1 = nn.Linear(embedding_size, num_tokens)
        self.device = device

    def forward(self, h_t, testing, corr_pct=0, eps=1e-8):
        batch_size = h_t.size()[0]

        h = h_t
        c = torch.zeros((batch_size, self.embedding_size), dtype=torch.float, device=self.device)

        last_token = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        utterance_nodes = []
        utterance = torch.zeros(batch_size, self.max_len, dtype=torch.long, device=self.device)
        entropy = 0
        matches_argmax_count = 0
        stochastic_draws_count = 0

        for i in range(self.max_len):
            embedded = self.embedding(last_token)
            h, c = self.lstm(embedded, (h, c))
            logits = self.h1(h)

            raw_noise = torch.randn(logits.shape, device=self.device).detach()
            noise_min = torch.min(raw_noise, keepdim=True, dim=1)[0]
            noise_max = torch.max(raw_noise, keepdim=True, dim=1)[0]

            min_val = torch.min(logits, keepdim=True, dim=1)[0]
            max_val = torch.max(logits, keepdim=True, dim=1)[0]
            noise = min_val + (max_val - min_val) * (raw_noise - noise_min) / (noise_max - noise_min)
            logits = (1 - corr_pct) * logits + corr_pct * noise

            probs = F.softmax(logits, -1)

            _, res_greedy = probs.detach().max(1)
            res_greedy = res_greedy.view(-1, 1).long()

            log_g = None
            if not testing:
                a = torch.multinomial(probs, 1)
                g = torch.gather(probs, 1, a.detach())
                log_g = g.log()
                a = a.detach()
            else:
                a = res_greedy

            matches_argmax = res_greedy == a
            matches_argmax_count += matches_argmax.int().sum()
            stochastic_draws_count += batch_size

            if log_g is not None:
                utterance_nodes.append(log_g)
            last_token = a.view(batch_size)
            utterance[:, i] = last_token
            probs = probs + eps
            entropy -= (probs * probs.log()).sum(1).sum()

        return utterance_nodes, utterance, entropy, matches_argmax_count, stochastic_draws_count


class ProposalPolicy(nn.Module):
    def __init__(self, device, embedding_size=100, num_counts=6, num_items=3):
        super().__init__()
        self.num_counts = num_counts
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.device = device
        self.fcs = []
        for i in range(num_items):
            fc = nn.Linear(embedding_size, num_counts)
            self.fcs.append(fc)
            self.__setattr__('h1_%s' % i, fc)

    def forward(self, x, testing, eps=1e-8):
        batch_size = x.size()[0]
        proposal_nodes = []
        entropy = 0
        matches_argmax_count = 0
        stochastic_draws_count = 0
        proposal = torch.zeros(batch_size, self.num_items, dtype=torch.long, device=self.device)
        for i in range(self.num_items):
            logits = self.fcs[i](x)
            probs = F.softmax(logits, -1)

            _, res_greedy = probs.detach().max(1)
            res_greedy = res_greedy.view(-1, 1).long()

            log_g = None
            if not testing:
                a = torch.multinomial(probs, 1)
                g = torch.gather(probs, 1, a.detach())
                log_g = g.log()
                a = a.detach()
            else:
                a = res_greedy

            matches_argmax = res_greedy == a
            matches_argmax_count += matches_argmax.int().sum()
            stochastic_draws_count += batch_size

            if log_g is not None:
                proposal_nodes.append(log_g)
            probs = probs + eps
            last_token = a.view(batch_size)
            proposal[:, i] = last_token
            entropy -= (probs * probs.log()).sum(1).sum()

        return proposal_nodes, proposal, entropy, matches_argmax_count, stochastic_draws_count


class AgentModel(nn.Module):
    def __init__(self, enable_comms, enable_proposal, device, term_entropy_reg, utterance_entropy_reg,
            proposal_entropy_reg, embedding_size=100):
        super().__init__()
        self.term_entropy_reg = term_entropy_reg
        self.utterance_entropy_reg = utterance_entropy_reg
        self.proposal_entropy_reg = proposal_entropy_reg
        self.embedding_size = embedding_size
        self.enable_comms = enable_comms
        self.enable_proposal = enable_proposal
        self.context_net = NumberSequenceEncoder(num_values=6, device=device)
        self.utterance_net = NumberSequenceEncoder(num_values=10, device=device)
        self.proposal_net = NumberSequenceEncoder(num_values=6, device=device)
        self.proposal_net.embedding = self.context_net.embedding
        self.device = device

        self.combined_net = CombinedNet()

        self.term_policy = TermPolicy()
        self.utterance_policy = UtterancePolicy(device)
        self.proposal_policy = ProposalPolicy(device)

    def forward(self, pool, utility, m_prev, prev_proposal, testing, corr_pct):
        batch_size = pool.size()[0]
        context = torch.cat([pool, utility], 1)
        c_h = self.context_net(context)
        if self.enable_comms:
            m_h = self.utterance_net(m_prev)
        else:
            m_h = torch.zeros(batch_size, self.embedding_size, device=self.device, dtype=torch.float)
        p_h = self.proposal_net(prev_proposal)

        h_t = torch.cat([c_h, m_h, p_h], -1)
        h_t = self.combined_net(h_t)

        entropy_loss = 0
        nodes = []

        term_node, term_a, entropy, term_matches_argmax_count = self.term_policy(h_t, testing=testing)
        nodes.append(term_node)
        entropy_loss -= entropy * self.term_entropy_reg

        if self.enable_comms:
            utterance_nodes, utterance, utterance_entropy, utt_matches_argmax_count, \
                                                utt_stochastic_draws = self.utterance_policy(h_t, testing=testing,
                                                                                            corr_pct=corr_pct)
            nodes += utterance_nodes
            entropy_loss -= self.utterance_entropy_reg * utterance_entropy
        else:
            utt_matches_argmax_count = 0
            utt_stochastic_draws = 0
            utterance = torch.zeros(batch_size, 6, device=self.device, dtype=torch.long)

        proposal_nodes, proposal, proposal_entropy, prop_matches_argmax_count, \
                                                prop_stochastic_draws = self.proposal_policy(h_t, testing=testing)
        nodes += proposal_nodes
        entropy_loss -= self.proposal_entropy_reg * proposal_entropy

        return nodes, term_a, utterance, proposal, entropy_loss, \
            term_matches_argmax_count, utt_matches_argmax_count, utt_stochastic_draws, prop_matches_argmax_count, prop_stochastic_draws
