# Momentum-SAM-ScheduleFree
Initial run showing we can find a very flat solution (sometimes) using MSAM and schedule free combined in [colab nb](https://colab.research.google.com/drive/1SWsIn1L2vc0AuWnAybO5pJmMGvPDE5Q6?usp=sharing)

![image](https://github.com/user-attachments/assets/67cd0cd4-305b-43aa-80a0-7babd42f09f4)

Note: Adam in above is actually Adamized Schedlue-Free Momentum-SAM 

This shows that Adamized Schedlue-Free Momentum-SAM usually finds worse solutions in terms of train and test cross entropy compared to [PSGD](https://github.com/lixilinx/psgd_torch/tree/master), but can often find flatter solutions than it. 

This might be due to the two backwards for every 1 backward allowed for PSGD. 

# Combined AdamW-MSAM-ScheduleFree Optimizer Formula 

## Parameters and Variables

For each parameter θ:

- x_t: momentum (used for MSAM ascent)
- m_t: first moment estimate (exp_avg)
- v_t: second moment estimate (exp_avg_sq)
- g_t: gradient at step t
- α_t: adaptive learning rate at step t
- β1, β2: exponential decay rates for moment estimates
- ε: small constant for numerical stability
- ρ: MSAM ascent step size
- λ: weight decay coefficient

## Adaptive Learning Rate (Schedule-Free Part)

α_t = lr * min(1, t / warmup_steps) * sqrt(1 - β2^t) / (1 - β1^t)

## Update Steps

1. Update moment estimates:
   m_t = β1 * m_t-1 + (1 - β1) * g_t
   v_t = β2 * v_t-1 + (1 - β2) * g_t^2

2. Compute MSAM ascent direction:
   d_t = x_t-1 / ||x_t-1||

3. MSAM ascent step:
   θ_t' = θ_t-1 + ρ * d_t

4. AdamW-style update:
   θ_t = θ_t' - α_t * m_t / (sqrt(v_t) + ε) - α_t * λ * θ_t'

5. Update momentum (x) for MSAM:
   x_t = β1 * x_t-1 + g_t

6. Schedule-free weighting:
   w_t = t^r * (max_α)^p  (where max_α is the maximum learning rate seen so far)
   c_t = w_t / Σ_i=1^t w_i

7. Combine schedule-free approach with MSAM:
   θ_t = (1 - c_t) * θ_t-1 + c_t * θ_t

## Complete Update Formula

The complete update for a parameter θ at step t can be summarized as:

θ_t' = θ_t-1 + ρ * x_t-1 / ||x_t-1||
θ_t'' = θ_t' - α_t * m_t / (sqrt(v_t) + ε) - α_t * λ * θ_t'
θ_t = (1 - c_t) * θ_t-1 + c_t * θ_t''

x_t = β1 * x_t-1 + g_t

Where the MSAM ascent is performed before the main update, and the momentum update follows.

## Key Components

This formula combines:
- The adaptive moment estimation from Adam
- The weight decay approach from AdamW
- The momentum-based sharpness-aware ascent step from MSAM
- The schedule-free adaptive learning rate and parameter interpolation


## Training code style 
   ```python
   optimizer.train()
   for epoch in range(num_epochs):
       for batch in dataloader:
           optimizer.zero_grad()
           
           # First forward-backward pass
           outputs = model(batch)
           loss = criterion(outputs, targets)
           loss.backward()
           
           # SAM-like ascent step
           optimizer.move_to_ascent()
           
           # Second forward-backward pass
           outputs = model(batch)
           loss = criterion(outputs, targets)
           loss.backward()
           
           # Move back and update
           optimizer.move_from_ascent()
           optimizer.step()
   
       # Evaluation
       optimizer.eval()
       # ... perform evaluation ...
       optimizer.train()
