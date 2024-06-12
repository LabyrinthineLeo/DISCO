import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu" if not torch.cuda.is_available() else "cuda"


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class SelfAttention(torch.nn.Module):
    def __init__(self, n_dim, n_heads=1):
        super(SelfAttention, self).__init__()
        self.n_dim = n_dim
        self.n_heads = n_heads
        self.query = torch.nn.Linear(n_dim, n_dim)
        self.key = torch.nn.Linear(n_dim, n_dim)
        self.value = torch.nn.Linear(n_dim, n_dim)

    def forward(self, x):
        bsz, seq_len, *rest = x.size()

        if len(x.shape) == 4:
            x = x.reshape(bsz * seq_len, -1, self.n_dim)  # Reshape to [bsz * seq_len, n, n_dim]
            n_dim = x.size(-1)

            # Linear transformation for query, key, value
            queries = self.query(x).reshape(bsz, seq_len, rest[0], self.n_heads, n_dim // self.n_heads).permute(0, 1,3,2, 4)
            keys = self.key(x).reshape(bsz, seq_len, rest[0], self.n_heads, n_dim // self.n_heads).permute(0, 1, 3, 2, 4)
            values = self.value(x).reshape(bsz, seq_len, rest[0], self.n_heads, n_dim // self.n_heads).permute(0, 1, 3, 2, 4)

            # Compute attention scores
            attention_scores = torch.einsum("bshnd,bthnd->bsthn", queries, keys) / (n_dim // self.n_heads) ** 0.5
            
            attention_weights = F.softmax(attention_scores, dim=-3)

            # Apply attention weights to values
            attended_values = torch.einsum("bsthn,bthnd->bshnd", attention_weights, values)

            # Reshape back to the original shape
            attended_values = attended_values.contiguous().reshape(bsz, seq_len, rest[0], n_dim)

        else:
            x = x.reshape(bsz * seq_len, self.n_dim)  # Reshape to [bsz * seq_len, n_dim]
            n_dim = self.n_dim

            # Linear transformation for query, key, value
            queries = self.query(x).reshape(bsz, seq_len, self.n_heads, n_dim // self.n_heads)
            keys = self.key(x).reshape(bsz, seq_len, self.n_heads, n_dim // self.n_heads)
            values = self.value(x).reshape(bsz, seq_len, self.n_heads, n_dim // self.n_heads)

            # Compute attention scores
            attention_scores = torch.einsum("bshd,bthd->bsth", queries, keys) / (n_dim // self.n_heads) ** 0.5
            attention_weights = F.softmax(attention_scores, dim=-2)

            # Apply attention weights to values
            attended_values = torch.einsum("bsth,bthd->bshd", attention_weights, values).reshape(bsz, seq_len, n_dim)
        

        return attended_values


class NGCF_DISCO(torch.nn.Module):
    def __init__(self, user_num, item_num, latent_dim, adj_norm, n_level, leaf_dim, beta=1.0, n_classes=1,
                 num_layers=2):
        super(NGCF_DISCO, self).__init__()
        self.model_name = "ngcf_disco"
        self.n_classes = n_classes
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout_rates = [0.1] * self.num_layers
        self.hidden_size_list = [latent_dim] * (self.num_layers + 1)
        self.hidden_size = 256
        self.beta = beta
        self.n_level = n_level 
        self.leaf_dim = leaf_dim
        self.eps, self.tau = 0.1, 0.2

        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)

        self.E_ego = None
        self.adj_norm = adj_norm 
        self.user_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(user_num, latent_dim)))
        self.item_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(item_num, latent_dim)))
        self.ego_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(user_num + item_num, latent_dim)))

        self.E_ego_list = [None] * (self.num_layers + 1)
        self.E_ego_list[0] = self.ego_emb
  
        for i in range(self.num_layers):
            self.GC_Linear_list.append(nn.Linear(self.hidden_size_list[i], self.hidden_size_list[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.hidden_size_list[i], self.hidden_size_list[i + 1]))
            self.dropout_list.append(nn.Dropout(self.dropout_rates[i]))

        self.user_hier_list = nn.ModuleList()
        self.item_hier_list = nn.ModuleList()

        self.user_disen_models = nn.ModuleList(
            [Encoder(self.latent_dim, self.hidden_size, self.leaf_dim) for i in range(self.n_level)])
        self.item_disen_models = nn.ModuleList(
            [Encoder(self.latent_dim, self.hidden_size, self.leaf_dim) for i in range(self.n_level)])

        for i in range(self.n_level):  
            self.user_hier_list.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),  
                    # nn.Sigmoid()
                    # nn.ReLU()
                    nn.Tanh()
                )
            )
            self.item_hier_list.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    # nn.Sigmoid()
                    # nn.ReLU()
                    nn.Tanh()
                )
            )

        # self-attention module
        self.SAM = SelfAttention(self.leaf_dim)

        # diagnosis module
        self.diag_layers = nn.Sequential(
            nn.Linear(3 * self.leaf_dim, self.latent_dim),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.diag_layers_sigmoid = nn.Sequential(
            nn.Linear(self.leaf_dim, self.latent_dim),
            nn.Sigmoid(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.predict = torch.nn.Linear(self.latent_dim, n_classes)
        self.logistic = torch.nn.Sigmoid()

    def cl_loss(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss_sum = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss_sum)

    def cl_module(self, input_embeds):
        loss_list = []
        for i in range(self.n_level):
            each_embed = input_embeds[i]
            random_noise = torch.rand_like(each_embed).to(device)
            each_embed_new = each_embed + (torch.sign(each_embed) * F.normalize(random_noise, dim=-1) * self.eps)
            loss = self.cl_loss(each_embed, each_embed_new, self.tau)
            loss_list.append(loss)
        return torch.mean(torch.Tensor(loss_list))


    def forward(self, user_id, item_id, hier_skills, test=False):
        if not test:  # train phase
            for layer in range(1, self.num_layers + 1):
                side_embeddings = torch.sparse.mm(self.adj_norm, self.E_ego_list[layer - 1])
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[layer - 1](side_embeddings))  
                bi_embeddings = torch.mul(self.E_ego_list[layer - 1], side_embeddings)  
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[layer - 1](bi_embeddings))

                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[layer - 1](ego_embeddings)  # dropout
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                self.E_ego_list[layer] = norm_embeddings  # append

            
            self.E_ego = sum(self.E_ego_list)
            E_u_norm, E_i_norm = torch.split(self.E_ego, [self.user_num, self.item_num], 0)

        else:
            E_u_norm, E_i_norm = torch.split(self.E_ego, [self.user_num, self.item_num], 0)

        user_embedding = E_u_norm[user_id]
        item_embedding = E_i_norm[item_id]

    
        Flag = False  
        if len(hier_skills.shape) == 3:
            Flag = True
        hier_skill_list = []
        for i in range(self.n_level):
            start_id = i * self.leaf_dim
            end_id = min(start_id + self.leaf_dim, hier_skills.shape[-1])
            if Flag:
                hier_skill_list.append(hier_skills[:, :, start_id:end_id].to(device))
            else:
                hier_skill_list.append(hier_skills[:, start_id:end_id].to(device))
       

        user_hier_reps = [None] * self.n_level  
        user_hier_recs = [None] * self.n_level  
        item_hier_reps = [None] * self.n_level  
        item_hier_recs = [None] * self.n_level  
       
        diag_list = []  
        for layer in range(self.n_level):  
            user_hier_reps[layer] = self.user_hier_list[layer](user_embedding)  
            item_hier_reps[layer] = self.item_hier_list[layer](item_embedding) 
        for layer in range(self.n_level):  
            h_u = self.user_disen_models[layer](user_hier_reps[layer])
            if not test:
                user_hier_recs[layer] = h_u
            else:
                user_hier_recs[layer] = h_u

            h_i = self.item_disen_models[layer](item_hier_reps[layer])
            if not test:
                item_hier_recs[layer] = h_i
            else:
                item_hier_recs[layer] = h_i

        
        cl_losses = 0.
        if not Flag:
            input_user_embeds = torch.stack(user_hier_recs, dim=0).permute(1, 0, 2)  
            input_item_embeds = torch.stack(item_hier_recs, dim=0).permute(1, 0, 2)  
            user_hier_recs_new = torch.unbind(self.SAM(input_user_embeds).permute(1, 0, 2), dim=0)  
            item_hier_recs_new = torch.unbind(self.SAM(input_item_embeds).permute(1, 0, 2), dim=0)  
        else:
            input_user_embeds = torch.stack(user_hier_recs, dim=0).permute(1, 0, 2, 3) 
            input_item_embeds = torch.stack(item_hier_recs, dim=0).permute(1, 0, 2, 3) 
            user_hier_recs_new = torch.unbind(self.SAM(input_user_embeds).permute(1, 0, 2, 3),
                                              dim=0)  # n_level*[bsz, n, n_dim]
            item_hier_recs_new = torch.unbind(self.SAM(input_item_embeds).permute(1, 0, 2, 3),
                                              dim=0)  # n_level*[bsz, n, n_dim]
        
        
        

        if not Flag:
            cl_losses = self.cl_module(user_hier_recs_new) + self.cl_module(item_hier_recs_new)
        else:
            cl_losses = 0.

        
        # Cognitive Diagnosis Module
        for i in range(self.n_level):
            diag = (nn.Sigmoid()(user_hier_recs_new[i]) - nn.Sigmoid()(item_hier_recs_new[i])) * hier_skill_list[i].to(torch.float32)
            diag_list.append(diag)
        cat_matrix = torch.cat(diag_list, dim=-1)
        output = self.diag_layers(cat_matrix)
        output = self.predict(output)
        output = torch.softmax(output, dim=-1)
        return output, cl_losses
    