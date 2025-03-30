'''
Model_dagger0
CW Cumulative CTE: 876.4440673713698
timesteps: 3462, expert_timesteps: 512
CCW Cumulative CTE: 1142.8016374197052
timesteps: 3345, expert_timesteps: 515

Model_dagger1
CW Cumulative CTE: 770.8276746860329
timesteps: 3522, expert_timesteps: 0
CCW Cumulative CTE: 1025.8597603614526
timesteps: 3527, expert_timesteps: 5

Model_dagger2
CW Cumulative CTE: 757.6815053488382
timesteps: 3522, expert_timesteps: 0
CCW Cumulative CTE: 925.0224176236867
timesteps: 3524, expert_timesteps: 0
'''
import matplotlib.pyplot as plt

cte_CW = [876.4440673713698, 770.8276746860329, 757.6815053488382]
pct_expert_CW = [1, 512.0 / 3462.0, 0.0, 0.0]
cte_CCW = [1142.8016374197052, 1025.8597603614526, 925.0224176236867]
pct_expert_CCW = [1, 515.0 / 3345.0, 5.0 / 3527.0, 0.0]

dagger_itr = ['No DAgger', '1 iteration of DAgger', '2 iterations of DAgger']

plt.plot(dagger_itr, cte_CW, marker='o', linestyle='-')
plt.title('Cumulative Cross-Track Error for Robot Moving Clockwise')
plt.ylabel('Cumulative Cross-Track Error (in/min)')
plt.show()

plt.plot(dagger_itr, cte_CCW, marker='o', linestyle='-')
plt.title('Cumulative Cross-Track Error for Robot Moving Counterclockwise')
plt.ylabel('Cumulative Cross-Track Error (in/min)')
plt.show()

