def run_episode(env,agent) :
    user,candidates=env.reset()
    agent.observe_candidates(user,candidates)
    done=False
    ep_reward=0

    while not done :
        action=agent.get_action(candidates)
        candidates, choice, reward, done=env.step(action)
        agent.observe(candidates,action,choice,reward,done)
        ep_reward+=reward
    return ep_reward

def run_train(env,agent,n_episodes) :
    agent.eval_off()
    for _ in range(n_episodes) :
        run_episode(env,agent)

def run_eval(env,agent,n_episodes) :
    agent.eval_on()
    rewards=[]
    for _ in range(n_episodes) :
        rewards.append(run_episode(env,agent))
    return rewards

def run_train_eval(env,agent,n_episodes_train,n_episodes_eval) :
    run_train(env,agent,n_episodes_train)
    return run_eval(env,agent,n_episodes_eval)