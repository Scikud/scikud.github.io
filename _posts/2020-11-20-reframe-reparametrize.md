---
layout:     post
title:      Reframe/Reparametrize 
date:       2020-11-22 12:56:29
summary:    Another post courtesy of ya boy Lagrange 
categories: OpenAI 
---

In continuation of the tradition of this blog, where I start off writing a post about one topic and ultimately end up deleting it a quarter of the way through and begin writing about something totally different, I present to you this week's topic: reframing and reparametrization using Lagrange multipliers. 

One tool i've found particularly useful throughout the years is taking problems I have in one domain and trying to reframe them as an optimization problem in order to use the method of Lagrange multipliers to arrive at a solution. The utility of [Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) (and more broadly [the KKT conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) ) is that as long as you can frame what you care about as some sort of constrained optimization problem, the method of lagrange multipliers gives you a simple but powerful method to derive algorithms as well as understand and bound your problem's sensitivites. I've always found this to be really cool, because at it's core you simply specify something you'd like to optimize (even if it's non-convex), specify contraints that thing should ideally satisfy and like magic, out pops an algorithm or a function corresponding to the optimum (or if not you usually at least get computable conditions that optimum should satisfy).

 I'll walk through 3 examples of this type of constrained optimization that have popped up for me over the past month in the context of machine learning. Namely, using the Lagrange formalism as one perspective on how to  arrive at the origins of the Gaussian distribution, the Backprop algorithm, and the Trust Region Policy Optimization (TRPO) algorithm in reinforcement learning (in some ways the predecessor of the more widely used PPO algorithm).


## Origins of the Gaussian (Maximum Entropy Distributions)
A natural question to ask is: What's the most general distribution one can use to describe random variables with finite mean and finite variance? 

The entire point of statistics is to infer both the shape and the parameters that control the underlying distributions generating our samples. What's the best we can do? Considering the space of all possible distributions we could potentially choose from we'd like to find a distribution that satisfies these constraints as weakly as possible. Having as little pre-existing structure as possible ensures it remains as general as possible, making the "least claim of being informed beyond the stated prior data". What we're really saying  is that we want a distribution that maximizes entropy[^1].

$$ \text{maximize}[H(x)] = \text{maximize}[\int_{-\infty}^{\infty}p(x)\log(P(x)dx ]$$.

To reiterate the distribution must also satisfy the finite mean and finite variance constraints, explicitly:

* **Constraint 0** :Probability density function must sum to 1:  $$\int_{-\infty}^{\infty}p(x)dx = 1$$ 
* **Constraint 1** :Finite mean:  $$\int_{-\infty}^{\infty}p(x)xdx = \mu $$
* **Constraint 2** :Finite variance: $$\int_{-\infty}^{\infty}p(x)\left(x-\mu\right)^2dx = \sigma $$ 

Putting in all together using the method of Lagrange multipliers we have:

$$
\begin{align}
\mathcal{L}= 
\int_{-\infty}^{\infty}p(x)\log p(x)dx 
-\lambda_0\left(\int_{-\infty}^{\infty}(p(x)-1\right)dx 
-\lambda_1\left(\int_{-\infty}^{\infty}p(x)(x)dx-\mu\right) \\
-\lambda_2\left(\int_{-\infty}^{\infty}p(x)(x-\mu)^2dx-\sigma\right) 
\end{align}
$$

Since $$\mathcal{L} = \mathcal{L}(P(x), \lambda)$$, taking the gradients we have :

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial p(x)} =  \log p(x) +1 - \lambda_0 - \lambda_1x -\lambda_2(x-\mu)^2 = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda_0}  = \int_{-\infty}^{\infty}p(x)dx-1 = 0\\\
\frac{\partial \mathcal{L}}{\partial \lambda_1} = \int_{-\infty}^{\infty}p(x)xdx-\mu = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda_2}  = \int_{-\infty}^{\infty}p(x)\left(x-\mu\right)^2dx-\sigma = 0 \\
\end{align}
$$

I won't put the algebra here (it's not at all interesting and you probably shouldn't much care) but you start with 

$$\log p(x) = \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2 -1 $$ 

keep cranking and you'll eventually end up with the following:

$$ P(x) = \frac{1}{\sqrt{2\pi \sigma^2}} exp\left(-\frac{(x-mu)^2}{2 \sigma^2}\right) $$

The Gaussian distribution at last. To recap, by framing our original question as a constrained optimization problem we were able to retrieve the gaussian distribution in a natural sort of way.

## Origins of Backprop 
It turns out we can reframe back-propagation in a [similar fashion](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf). Ultimately, supervised learning is about minimizing the error between some labeled data $$Y$$ and our model's predictions, usually given by the last layer of our neural network $$Z^L$$. 

Compactly: minimize $$ Loss(z^L, y) $$, where $$ z^{i} = f^i(z^{i-1},W^i) $$

At it’s core, it’s just another instance of a constrained optimization problem that can be tackled with the same Lagrangian formalism as before.

$$\mathcal{L}(z, W, \lambda)=  Loss(z^L, y) -\sum\limits_{i=1}^L \lambda_{i}^T\left(z^i - f^i(z^{i-1})\right)  $$

Above, we express the dynamics of each layer of the network as a sum through all $$L$$ layers. Effectively, all we're saying is --minimize some loss function  *S'il vous plaît* --with the constraint that each layer $$ Z^i$$ of our network is just a function of the previous layer $$Z^{i-1}$$ through some non-linear function $$f$$. 

Taking the gradient we arrive at the following:

$$
\begin{align}
\nabla_\boldsymbol{\lambda_i} \mathcal{L} = z^i - f^i(z^{i-1},W^i) = 0 \\
\nabla_\boldsymbol{w} \mathcal{L} = -\sum\limits_{i=1}^L \lambda_i^T \nabla_W f^i(z^{i-1},W^i) = 0 \\ 
\nabla_\boldsymbol{z^i} \mathcal{L} = \lambda_i - \nabla f^{i+1}(z^i,w^{i+1}) = 0 \\
\nabla_\boldsymbol{z^L} \mathcal{L} = \lambda_L - \nabla Loss(z^L, y) = 0 \\
\end{align}
$$

Taking a looking at above, one gets almost for free the following: 
$$ z^i =f^i(z^{i-1} $$ and $$ \lambda_L = \nabla_{z^L} Loss(z^L,y)$$

Additionally, Working through these (particularly the third equation)  one eventually arrives at

$$\lambda_j = \sum_{i \in \beta(j)} \lambda_i \frac{\partial f_i (z^i, W^i)}{\partial z^j} $$  

where $$ \beta(j) $$ is the set of all incoming edges from the vertex j[^2]

Which is recognizable in the context of backprop as the equation for the the adjoints $$\lambda_ i$$, telling us how to measure the sensitivity of one node relative to the previous layer.  Cool.
## Origins of TRPO
Man, bad news homie -- this post is already much longer than I originally intended (and I'm getting *real* tired to typing out LaTex), so maybe I won't go into as much detail about this one. Anyway, It turns out one can arrive at the core of [TRPO](https://arxiv.org/pdf/1502.05477.pdf) algorithm in reinforcement learning under a similar scheme. I'll try to lay out the crux briefly:

[Reinforcement learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html) fundamentally differs from supervised learning in some important ways. Particularly, in the supervised learning case your samples are drawn i.i.d from some underlying distribution and therefore it ultimately doesn't really matter if you sample from a "bad" (low reward) region of your parameter space in one batch, because your next batch will not be conditioned on the poor performing batch in any way. The issue in reinforcement learning[^3] is that this sampling i.i.d assumption no longer holds, precisely because your policy ultimately also *controls* your sampling process; your policy both learns from your samples and is responsible for gathering new samples.

Ultimately we'd only like to trust our policy within some small region. Ostensibly, a natural way to do this is to bound our parameter updates such that they stay within some small radius of their original values.  

$$\theta' \leftarrow  \arg\max_{\theta} (\theta' - \theta)^T \nabla J(\theta) $$

$$ s.t \quad ||\theta' - \theta || ^2 \leq \epsilon $$


In reality, however we don't have much reason to expect this to be meaningful since some parameters might change much more quickly or slowly than others. Ultimately what we care about isn't *really* that the parameters that control out policy be bounded from one update to the next, it's that change in the policy *itself* be bounded. 

We'd like to find a way to restrict our policy updates that isn't explicitly dependent on whatever parametrization we used. One way to do this is instead of restricting the gradients, we  should restrict the [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the policy and it's update to be bound by some epsilon ball. Sure sounds like a constrained optimization problem don't it? 😉


---
[^1]: https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution
[^2]: Worked out in more detail [here](https://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/) for the interested
[^3]: At least in the on-policy case