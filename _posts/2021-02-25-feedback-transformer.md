---
layout:     post
title:      Crunchtime
date:       2021-02-25 12:56:29
summary:    Sharing Some Code
categories: OpenAI
---


### Utility functions
I've spent the past week writing some desperately neeeded utility functions to instrument the code that i've built out as part of the scholar's program. Largely, i've been trying to gain a better understanding of exactly what's going on inside these models as they operate over data continously. One lesson learned is on the importance of writing utility functions early, not only because it saves you valuable time when approaching deadlines, but also because being really explicit up front about the kind of data you would like to collect also informs what kind of experiments your'e likely to run. 

### Feedback Transformer
Last blog post, I mentioned I've been playing around with implementing the feedback transformer. The principle idea in the feedback transformer is to allow low level representation in transformers to attend to previous higher level representations. This modifies the computational path of the the traditional transformer architecture and transforms it something functionally resembelling an autoregressive RNN. I wrote out a quick and dirty implementation for this (below), which I'll clean up and post on github at some point, along with the million other things I have to catch up on.  

~~~ javascript
class MultiHeadedAttn(nn.Module):
    def __init__(self, Config):
        super(MultiHeadedAttn, self).__init__()
        self.c = c = Config
        # Generates queries, keys, values
        self.fc1 = nn.Linear(c.embdSize, (c.qkvSize)*c.numAttnHeads)
        self.fc2 = nn.Linear(c.embdSize, c.embdSize)
        # Create Mask
        self.register_buffer(
            "causalMask",
            torch.tril(torch.ones((c.blockSize, c.blockSize))))
        self.register_buffer(
            "padMask",
            torch.ones(c.blockSize, c.blockSize))

    def forward(self, x, k, v):
        B, T, embdSize = x.shape  # B =Batch size,  T = numTokens
        h = self.c.numAttnHeads

        q = self.fc1(x)
        q = q.reshape(B,h,T,-1)
        k = k.reshape(B,h,T,-1)
        v = k.reshape(B,h,T,-1)
        # God bless einsum
        attn = torch.einsum('bhij,bhkj->bhik', q, k)
        mask = torch.unsqueeze(self.padMasks, 1).repeat(
            1, h, 1, 1)[..., :T,:T] *self.causalMask[:T, :T]
        attn = attn.masked_fill(mask == 0., float('-inf'))
        scores = F.softmax(attn/np.sqrt(self.c.qkvSize), -1)
        outpt = torch.einsum('bhij,bhjk->bhik', scores, v)
        outpt = outpt.view(B, T, embdSize)

        return self.fc2(outpt)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        c = config
        self.ln1 = nn.LayerNorm(c.embdSize)
        self.attn = MultiHeadedAttn(config)
        self.mlp = nn.Sequential(
            nn.Linear(c.embdSize, c.embdSize*4),
            nn.GELU(),
            nn.Linear(c.embdSize * 4, c.embdSize)
        )
        self.ln2 = nn.LayerNorm(c.embdSize)

    def _setPadMasks(self, padMasks):
        self.attn.padMasks = padMasks
        
    def forward(self, x, k, v):
        x = self.ln1(x+self.attn(x,k,v))
        x = self.ln2(x + self.mlp(x))
        return x


class TinyFeedbackTransformer(nn.Module):
    def __init__(self, Config):
        super(TinyFeedbackTransformer, self).__init__()
        # Size assertions
        assert Config.embdSize >= Config.numAttnHeads

        # Configuration stuff
        c = Config
        c.qkvSize = c.embdSize // c.numAttnHeads
        self.config = c

        self.wordEmbedding = nn.Embedding(
            c.paddingIndx+1, c.embdSize, padding_idx=c.paddingIndx)
        self.posEmbedding = nn.Parameter(
            torch.zeros(1, c.blockSize, c.embdSize))

        #New learnable parameters for feedback transformer
        self.memoryCoeff = nn.Parameter(torch.ones(1, c.numLayers))
        self.ffkv = nn.Linear(c.embdSize, (c.qkvSize*2)*c.numAttnHeads)

        self.blocks = nn.ModuleList(
            [DecoderBlock(c) for _ in range(c.numLayers)]
        )
        self.ln1 = nn.LayerNorm(c.embdSize)
        self.head = nn.Linear(c.embdSize, c.paddingIndx, bias=False)

        self.apply(self._init_weights)

    def forward(self, indxs, padMasks):
        for mod in self.blocks:
            mod._setPadMasks(padMasks)
        numTokens = indxs.shape[1]


        # Combine word and position embeddings
        x = self.wordEmbedding(indxs)
        pos = self.posEmbedding[:, :numTokens, :]
        x = x+pos

        #Initalize the memory tensor
        memory = torch.tensor([]).to(device)
        batchSize = x.shape[0]        
        blockSize = self.config.blockSize
        maxMemSize = self.config.memorySize
 

        finalOutputs = torch.tensor([]).to(device)
        #Pass through the transformer
        for indx in range(x.shape[1]):
            currSlice = x[:, indx, ...].view(batchSize,1,-1)
            inpt = torch.cat((memory,currSlice), dim=1)
            inpt = inpt[:, -maxMemSize:, ...]

            # Grab the W_k and the W_v parameters
            wk, wv = self.ffkv(inpt).chunk(2, 2)

            outputs = torch.tensor([]).to(device)
            for decoderBlock in self.blocks:
                inpt = decoderBlock(inpt, wk, wv)
                outputs = torch.cat((outputs, torch.unsqueeze(inpt[:,-1,:],0)))
            currmemory = torch.einsum('il,lbd->bd',
                                      torch.softmax(self.memoryCoeff, -1), outputs)
            
            memory = torch.cat((memory, torch.unsqueeze(currmemory,1)), dim=1)
            memory = memory[:, -maxMemSize:, ...]
            finalOutputs = torch.cat((finalOutputs,torch.unsqueeze(inpt[:,-1,:],1)), dim =1)
        finalOutputs = self.head(self.ln1(finalOutputs))

        return finalOutputs

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            d = (module.embedding_dim)**(1/2)
            module.weight.data.normal_(mean=0.0, std=0.125/d)
        if isinstance(module, nn.Linear):
            d = (module.in_features)**(1/2)
            module.weight.data.normal_(mean=0.0, std=0.125/d)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
~~~

Anywho, that's it for now. More to come in the coming weeks if I ever find time to organize my thoughts and my work. TTYL
