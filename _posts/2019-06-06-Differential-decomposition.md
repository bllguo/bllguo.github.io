---
layout: post
title: Due-tos
---

This post is a throwback to the methodology behind one of my first analytics projects at System1. The due-to is a simple name for a simple idea - isolating the effects of individual key performance indicators (KPIs) on a business metric, like gross profit. Sometimes - most times, even - data science doesn't have to be that sophisticated.

## 1. Problem Setup

Let us consider a simple example. Imagine being the proud owner of Widgets Incorporated. Your new widget has just entered the market and you want to do some digital marketing. 
1. You place an ad for your widget via Google Ads. After spending some amount at the auction, you end up acquiring some number of impressions, or some number of appearances on Google search engine result pages. A useful KPI here is the cost per impression, which we will denote hereafter as CPC. (Typically, in digital advertising, this is formulated as the cost per thousand impressions, or the cost per mille (CPM))
2. A few users who see your ad end up clicking through to your website, which extolls the virtues of the Widget 3.0. The number of clicks over the number of impressions will be the click-through rate, CTR.
3. From there, some number of visitors who click through also complete a purchase. Revenue is generated. We can easily define revenue per click, RPC.

Now, what is your gross profit $\pi$?

$$\pi = Revenue - Cost$$

Let us do some algebraic manipulations to define profit in terms of our KPIs:

$$\pi = Impressions(\frac{Revenue - Cost}{Impressions})$$

$$\pi = Impressions(\frac{Revenue}{Clicks}\frac{Clicks}{Impressions} - \frac{Cost}{Impressions})$$

$$\pi = Impressions(RPC\times CTR - CPC)$$

Let $V$ be the total volume of impressions.

$$\pi = V(RPC\times CTR - CPC)$$

You can monitor the performance of your ad looking at the gross profit as well as the daily values of your KPIs. 

Imagine now that you notice a drastic decrease in profit week-over-week, with corresponding changes in the key metrics. How can you understand more quantitatively the effect of V vs. RPC or CTR on the change in profit? What is the effect on profit *due to* changes in RPC or CPC?

## 2. Decomposition w/ Differentiation

More formally, we observe, over two time periods $t_1, t_2$, a change in $\pi$, $\Delta\pi$. 

$$\Delta\pi = \pi_{t_2} - \pi_{t_1}$$

We would like to identify how much of this $\Delta$ is due to the change in volume $\Delta V$, or change in RPC $\Delta RPC$, etc.

Differentiate!

$$\frac{\partial\pi}{\partial t} = \frac{\partial(V(RPC\times CTR - CPC))}{\partial t}$$

An elementary application of the chain rule will give us:

$$\frac{\partial\pi}{\partial t} = \frac{\partial V}{\partial t}(RPC\times CTR - CPC) + \frac{\partial RPC}{\partial t}(V\times CTR) + \\\frac{\partial CTR}{\partial t}(V \times RPC) - \frac{\partial CPC}{\partial t}(V)$$

We can approximate the partials with our observed deltas, using finite differences. Here we simply use the forward difference:

$$\frac{\Delta\pi}{\Delta t} = \frac{\Delta V}{\Delta t}(RPC\times CTR - CPC) + \frac{\Delta RPC}{\Delta t}(V\times CTR) + \\\frac{\Delta CTR}{\Delta t}(V \times RPC) - \frac{\Delta CPC}{\Delta t}(V)$$

Along the same vein, we can substitute in values for $V, RPC, CTR, CPC$ from our real observations. $V = V_{t_2}$, $V = V_{t_1}$, or $V = \frac{V_{t_2} + V_{t_1}}{2}$ are all valid choices.

So we have successfully decomposed $\Delta\pi$! Nota bene:

$$\Delta \pi = \Delta V \Delta \pi_{V} + \Delta RPC \Delta \pi_{RPC} + \Delta CTR \Delta \pi_{CTR} + \Delta CPC \Delta \pi_{CPC}$$

$$\Delta \pi_V = \Delta V(RPC\times CTR - CPC)$$

$$\Delta \pi_{RPC} = \Delta RPC(V\times CTR)$$

$$\Delta \pi_{CTR} = \Delta CTR(V \times RPC)$$

$$\Delta \pi_{CPC} = -\Delta CPC(V)$$

That is, the contribution to $\Delta \pi$ from each KPI is given directly from these terms. Dimensional analysis confirms that $\Delta \pi_{KPI}$ are all in units of $\Delta \pi$.

With this information we can build a slick waterfall chart that satisfies our equation, with the first four bars summing to the last (count the heights!).

![dueto]({{ site.baseurl }}/images/dueto.png)

In this case, reductions in cost per click since the last week drove the majority of the profit increase, but click-thru rate plummeted. This may indicate that the keywords we are bidding on are no longer very relevant. While competition in the auction is lower for these keywords, consumer interest has also waned to a degree that nearly offsets that change. We can extract clear practical insights immediately.

In general, this analysis can be useful in:
1. identifying which KPI has the most impact on our metric of interest
2. providing a bird's-eye view of macro-level trends and performance over two time periods
3. disentangling the effects of individual KPIs for particularly convoluted metrics