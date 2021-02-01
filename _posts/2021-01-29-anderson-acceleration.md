---
layout:     post
title:      On Fixed points
date:       2021-01-29 12:56:29
summary:    A quick point
categories: OpenAI
---

### Some thoughts
This week, i've been building out an implementation of a deep equilibrium model as a means to explore some of the ideas centering around test time compute. I've talked about deep equilibrium models[^1] briefly in [another post](https://scikud.github.io/openai/2020/12/20/troubles/), so I won't spend too much time here going over them, but generally the thesis is that any multilayer neural network could be represented by a single layer weight tied recurrent net and that if we perform repeated forward iteration for most "real" architectures this recurrent weight tied net tends to converge to a fixed point. It's an interesting idea, though the universality argument seems fine the expressivity and representational power of the fixed point seems to be a little dubious. 

My interest in deep equilibrium models stems from the idea that in some sense they represent the infinite depth limit of a neural network. This makes them seemingly natural places to study the limits of test time compute. But more on that in another post. For now i'll leave you with two things:


## Thing 1

The first thought comes from thinking more about the nature of the fixed point that deep equilibrium models converge to. Something seems a little funky about this (though admittedly this is likely just a failure of intuition rather than any real failing of the paradigm). Intuitively it would seem that forcing models to converge to some fixed point limits their expressivity. This isn't a formal argument by any means, but loosely we can imagine the only networks that converge to a fixed point are those with spectral norm < 1 i.e contraction mappings that converge as a consequence of Banach's fixed point theorem. While because of normalization I imagine most layers empirically satisfy this, it's not too much of a leap to imagine and construct layers for network where this is not the case. Even if we consider the contraction mapping case another interesting question to ask would be if there are instances where we're interested in some point that actually doesn't represent a fixed point but rather is some intermediary.


In practice we observe that while deep equilibrium models are much more memory efficient that traditional architectures there still exists a question of why their performance isn't dramatically better if they truly represent this "infinite depth limit". Maybe the reason lies at least partially with the limitations of restricting ourselves to only the fixed point. 


## Thing 2

The second thing i'd like to leave you with is this cool, (new to me) fixed point solver method called Anderson Acceleration. In traditional fixed point stuff you're trying to solve a  $$ f(c) = c $$ type equation. The boring way is to just choose some initial point $$ x_0 $$ and continue iteration with $$x_{i+1} =f(x_0) $$ until you find some point where the residual $$ x_{i+1} - x_{i}  $$ is lower than some tolerance threshold or you've maxed out your number of allowed iterations.

This is fine if you're lame and you're function isn't computationally expensive. But if iterating your function is itself a pain you'd like to find some way to accelerate the convergence to a fixed point. You could use Newton's method which certainly has faster convergence, but then you have to compute Jacobians. However, if you're function is computationally expensive, evaluating Jacobians is going to be computationally prohibitive as well. Again no fun. 

Apparently, all the cool kids have been using something called "Anderson Acceleration" which aims to accelerate the convergence of fixed point iteration. At its core its' basically like a finite difference secant method.  Instead of just using the previous guess to compute the next guess, we instead take a linear combination of the past $$ m $$ points. That is 

$$ x_{k+1} = \sum^{m_k}_{i=0}(\alpha_k)_if(x_{k-m_k+i}) $$

With the constraint that we would just like for these alphas to minimize the residuals over the past m iterations. That is, if we define the past m residuals to be 

$$ G = [f(x_k)-x_{k} ... f(x_{k-m+1}) -x_{k-m+1} ] $$ 

Minimize $$\| G\alpha\|^2_{2} $$ subject to the following normalization condition $$ 1^T\alpha = 1 $$ 

Which you can transform into a linear system and solve any way you like. 

The cool thing is that apparently in most cases Anderson acceleration not only accelerates convergence it also tends to avoids solution divergence.

Anywho, that's it for now. CYOTF ✌️

---
[^1]: https://arxiv.org/pdf/1909.01377.pdf