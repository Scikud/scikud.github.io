---
layout:     post
title:      Updating my views
date:       2021-02-12 12:56:29
summary:    idk man, stuff
categories: OpenAI
---
 
 
This past week, i've been playing around with implementing a feedback transformer as well as improving some of the baselines i've constructed. Lots of ongoing work on those fronts I don't particularly feel like writing about now. While explaining some of the things I've been working on to someone recently, I realized the way I think about the directionality of machine learning in aggregate has changed. The following post is largely a reconstruction of that conversation.
 
## The current state of affairs
Let's say we'd like to create a machine learning algorithm, particularly a neural network that learns some specific algorithm -- say Dijkstra's. Now, if the idea of creating a neural network to haphazardly implement a shitty approximation of a well known existing algorithm seems contrived,that's because it is- but stick with me I'm making a rhetorical point. Anyway, in order to carry out this task in the modern machine learning paradigm one is resigned to the somewhat tedious task of either finding or creating datasets that are sufficiently large and sufficiently expressive in the hopes of capturing enough complexity for your neural network to learn the algorithm you care about.
 
Paired with the increasing ubiquity of machine learning models[^1] themselves this seems to shift the task of learning the algorithm largely into the domain of data set engineering, i.e more specifically the problem of gathering and altering datasets ([Andrej's famous Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35)). In practice however, we usually find that the model finds it much easier to learn features specific to it's training dataset rather than general features of the problem domain you actually care about.  Again we run into the orignal issue; in order to even learn a relatively straightforward general algorithm we're forced to  either hamfist what we believe to be good inductive priors into our model or find/create exquisitely curated datasets that capture all the nuance we would like our model's outputs to capture. In the context of the current machine learning paradigm, for all interesting problems it seems exceedingly unlikely that such curation is possible; there seems to be some critical *general* ingredient missing. 
 
### What I thought would work
In response to this, my initial thoughts around test time compute went something like this : Maybe by more explicitly baking in compute at test time (baking in recurrence back in transformers etc), we could make it easier for our models to favor leveraging the additional computation over simply recognizing spurious patterns in the dataset. The results of the (admittedly naive) experiments I've been running are still TBD but in general the trend seems to be that this only helps in particular contexts, and even then the benefits aren't super pronounced.
What does seem a little clear now is that what I've been doing thus far amounts to something like a fancy form of regularization.
 
### Score models?
The problematic bit seems to be that using more compute at test time is only useful if your network's learning procedure actually is able to imbue it with some mechanism to make efficient use of this additional computation.  One way to make this  explicit is to incorporate a discriminator into our generative mechanism. At the beginning of this project, I chose to not pursue this direction because at the moment it didn't seem sufficiently general. Now however, the question that keeps popping up in the back of my head is whether it's impossible to actually construct a model that continually refines the quality and fidelity of its outputs without it also implicitly learning something roughly isometric to a discriminator or some sort of energy minimization mechanism.
 
### What does the future hold
I'm not sure at all where to go with all of this. The asymmetry between how comparatively easy it is to recognize a good solution to some problem and the difficulty in actually generating these solutions would seem to suggest we invest our efforts into mechanisms that learn score like correspondences between inputs and outputs and then deploy search mechanisms at test time to steer themselves to  increasingly more fit outputs.  Importantly, the idea i'm trying to convey here is that if what we care about is learning how to reason, rather than learning how to solve particular datasets then  what we should care about is not the outputs themselves  but rather the search mechanisms used to operate over this score landscape. Anyway, that's it for now. TTYL ✌️

 
---
[^1]: https://arxiv.org/abs/1706.05137

