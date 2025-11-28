In Stage 2 of the capstone project, you’ll take on a real-world style challenge in ML: optimising unknown functions using limited information. Starting with this module and continuing through to Module 24, you'll follow a process that reflects how optimisation is approached in research and industry – balancing exploration, evidence and strategy. By the end of Module 24, your submission should be complete. Module 25 gives you time to fine-tune your work and, if you choose, share your results.

Let’s walk through how the challenge works and what's expected of you.

### What is this challenge about?
This capstone project mimics a Bayesian optimisation-style competition, in which you will try to find the maximum of eight unknown functions, also known as black-box functions. You won't have the equations or visuals of these functions up front – just some initial data and the ability to make smart guesses.

Each function simulates a real-world task such as radiation detection, robot control or drug discovery, where evaluations are expensive or limited. This is your opportunity to apply what you’ve learned about intelligent search strategies to a realistic ML challenge. Perfect solutions are not expected. You’re encouraged to demonstrate a sound trial-and-error process throughout the capstone project.

### What will you do?
You’ll be working with eight synthetic black-box functions. These are unknown mathematical functions that accept inputs and return a single output. Your goal is to find the inputs that give the highest possible output for each function.

Every function is a maximisation problem. You won't see the internal workings of the functions, but you’ll be able to observe how they respond to different inputs.

Remember, each function is:

- A maximisation task 
- Initially represented by ten known data points
- Of increasing dimensionality (from 2D to 8D)

#### Each week, you will:

- Review your existing data that grows week by week
- Choose a new input point to query per function
- Submit your chosen input via the capstone project portal in the correct format
- Get the new output once the submission is processed
- Update your data set and revise your strategy
- After each round, be encouraged to reflect on:
- What method you used and why
- Whether your query focused on exploration or exploitation
- What the latest result taught you
- What your strategy is for the next round
- Note: You'll repeat this process each week, building your understanding iteratively, just as optimisation unfolds in real-world ML projects. Your weekly reflections will support your final write-up or presentation

Let’s look at an example to illustrate the process you’ll follow.

Suppose you’re working on Function 1, a 2D function. The data you start with might look like the following:
```
x: [0.1, 0.4], y: [0.45]
x: [0.2, 0.8], y: [0.61]
... (ten data points in total)
```
You then submit an input in this format: 
`0.123456-0.654321`

Later in the week, you’ll receive a new output `y` for this point, growing your data set to 11 points.

### What if you don't find the best result? 
That's okay! For example, one of the functions has two optima, but most will only find one. That’s still a strong result. In real-world ML, most progress is made not through perfect solutions but through thoughtful iteration and practical reasoning.

Note: make sure to submit your query and complete the required reflection each week. It’s your weekly reflection on strategy and next steps that counts towards the completion of this programme.

### What techniques can you use?
You're free to use any ML method to decide what input to query next. Here are some examples:

- Random search using np.random.uniform
- Grid search to evaluate points on a grid, such as 10,000 points
- Bayesian optimisation using a GP and acquisition functions such as UCB
- Manual reasoning using scatter plots or your own insight to pick the next best point
- Custom surrogate models to train an ML model to predict and guide your query
- A note on what's NOT required

In Stage 2 of the capstone project, you do not need to:

- Write and submit an optimiser
- Build a full optimisation model from scratch
- Find the perfect global maximum for every function
- Your task is to work with limited data, make smart decisions and reflect on your approach as you go.

### Summary
Here's how the capstone project works:

- You're maximising eight unknown functions, one query per function per week.
- You choose the ML method, such as random, grid, Bayesian optimisation or manual.
- You submit your inputs in a precise format via the capstone project portal.
- You reflect, revise and iterate over several rounds.
- Your success isn't just the highest value – it’s showing thoughtful, data-driven decision-making in your reflection.