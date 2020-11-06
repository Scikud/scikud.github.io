---
layout:     post
title:      OpenAIs Scholars Initial Thoughts
date:       2020-10-22 12:56:29
summary:    New blog who dis?
categories: OpenAI pixylls jeykll
---

I'm starting a new blog[^1]  to mark the beginning of my time as part of the OpenAI scholar's program

## Things I'm excited about
I'm deeply excited and humbled to work with such an incredible and deeply talented pool of individuals, both within my cohort and within OpenAI in general. The first week has been such a whirlwind of excitement. I've spent much of it shoring up my foundations in machine learning theory and application as well tyring to find my bearings in the wide sea of research interests.

I joined this program because of my belief that well aligned machine intelligence can act as an incredible force multiplier for a wide range of human endeavors. I'm most excited by the incredible generality of deep learning; the remarkable ability to abstract and  apply the same set of algorithms to a wide range of seemingly disparate problems. This paradigm excites me because it  reimagines computers and computation as being more than numerical automata but as robust tools capable of ingesting and producing rich inputs and outputs allowing us to augment our own problem solving capacity and by extension amplify ingenuity and improve the human condition. Honestly, what could be more exciting or compelling as a research interest?

## Ramblings 
One thing I've personally found useful is keep running track of ideas that emerge during the course of self study. I've found this idea tracking helpful in the meta-sense; in that, over time it helps me see emerging trends in my train of thought. I'm really not sure how useful these stream of consciousness type expositions will be for others, but nevertheless here are some of my thoughts from this week:

1. Training a neural network produces a model that we can think of as some object whose weights and biases provide discrete estimates of the high dimensional manifold our training data lies on. Along this train of thought, can we  think of the hyperparameters of two different networks trained on the same data as two different samples of the same underlying 'true' manifold? What then is the relationship (if any) between the weights of these two models? Is it possible to be clever and somehow combine these two manifold samplings to get a better estimate of the true manifold ala something loosely analogous to [Richardson Extrapolation](https://francisbach.com/richardson-extrapolation/) /Bayesian updating ?


2. When training neural networks we typically include as part of the loss function, a regularization term. The idea behind this is that penalizing large model weights we can prevent the co-adaptation of network weights and limit the networks ability to learn local noise in the underlying training data thus leading to better generalization.  The regularization terms i've seen thus far are fairy straightforward l1 or l2 norms on the network weights. I'm curious to explore the effectiveness of regularizing with more sophisticated techniques. Like regularizing with the goal to minimize entropy or something along that vein.

That's it for now. Catch you all later!



---
[^1]: For the literally two people who read the single post in the old blog, I've migrated that post as well :)