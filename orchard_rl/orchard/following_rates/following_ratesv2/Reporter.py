import math
import random

# from orchard_rl.orchard.actor_critic.policy_network import PolicyNetwork 


class Reporter:
    agent_id: int
    agents: list[int] # list of all agent ids

    budget: float 
    epsilon: float # epsilon <= budget / (number of actors - 1)

    # Following rates: (actor id) -> mu_{i,j}
    # mu_{i,j} = 0 if i = j
    following_rates: dict[int, float]

    # Expected Rewards: actor id -> estimated expected reward that 
    # agent j perceives from agent i's action
    expected_rewards: dict[int, float]

    # Probability of an actor acting
    # Initialized as uniform distribution for round-robin ordering
    acting_probabilities: dict[int, float]

    # Step size for learning expected rewards
    alpha_U: float 

    # Step size for learning following rates
    alpha_mu: float


    def __init__(self, agent_id: int, agents: list[int], budget: float, epsilon: float) -> None:
        self.agent_id = agent_id
        self.agents = agents

        self.budget = budget
        self.epsilon = epsilon

        # Initialize following rates to B / (N-1) for each actor-observer pair
        # Initialize estimated expected rewards to 0
        # Initialize acting probabilites to 1/N (uniform)
        self.following_rates = {}
        self.expected_rewards = {}
        self.acting_probabilities = {}
        for id in agents:
            if id != self.agent_id:
                self.following_rates[id] = (self.budget / (len(self.agents) - 1))
            else:
                self.following_rates[id] = 0
            self.expected_rewards[id] = 0
            self.acting_probabilities[id] = 1/len(self.agents)

        self.alpha_U = 0.1
        self.alpha_mu = 0.01


    def get_reporting_probability(self, actor_id: int) -> float:
        # Return the probabilty of observer reporting to actor
        following_rate = self.following_rates[actor_id]

        return 1 - math.exp(-following_rate) # q(mu_{i,j}) = 1 - exp(-mu_{i,j})

    def _is_reporting(self, actor_id: int) -> bool:
        # Return true if observer is reporting to actor, false if not
        return random.random() < self.get_reporting_probability(actor_id)
    

    def update(self, actor_id: int, perceived_reward: float):
        Uhat_j = self.expected_rewards[actor_id]
        mu_j = self.following_rates[actor_id]

        # update estimated expected rewards
        # Uhat_{i,j} <- Uhat_{i,j} + a_U * (r_i - Uhat_{i,j})
        self.expected_rewards[actor_id] += self.alpha_U * (perceived_reward - Uhat_j)


        # update following rates
        # mu_{i,j} <- mu_{i,j} + a_mu * ( q'(mu_{i,j}) / q(mu_{i,j}) ) * Uhat_{i,j}
        self.following_rates[actor_id] += \
        self.alpha_mu * math.exp(-mu_j) * Uhat_j / (1-math.exp(-mu_j))

        # Project
        self.following_rates = self._project(self.following_rates)

    
    def _project(self, fr: dict[int: float]) -> dict[int, float]:
        """
        Project following rates for observer
        """
        if sum(fr.values()) <= self.budget:
            return fr

        # 1. Shift vector by subtracting the floor
        shifted_fr = {k: v - self.epsilon for k, v in fr.items()}
        
        effective_budget = self.budget - (len(fr) * self.epsilon)
        
            
        # 3. Apply standard euclidean projection on the shifted values
        u = sorted(shifted_fr.values(), reverse=True)
        cssv = 0.0  
        theta = 0.0 
        
        for i, val in enumerate(u):
            cssv += val
            t = (cssv - effective_budget) / (i + 1)
            
            if val > t:
                theta = t
            else:
                break
                
        # 4. Subtract the threshold, clip to 0, and add the floor back
        for k, v in shifted_fr.items():
            shifted_fr[k] = max(v - theta, 0.0) + self.epsilon
        return shifted_fr

    # Main reporting method
    def get_report(self, actor_id: int, perceived_reward) -> float:
        # If observer is reporting to actor, return the observer's perceived_reward
        # If actor and reporter are the same agent, then return 0
        # Otherwise, return 0

        if self._is_reporting(actor_id) and not actor_id == self.agent_id:
            self.update(actor_id, perceived_reward)
            return perceived_reward
        return 0
    

if __name__ == "__main__":
    b = 10 
    e = 1
    r = Reporter(1, [], b, e)
    print(r._project({1: 1.5, 2: 2, 3: 1000}))


