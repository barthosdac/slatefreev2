import numpy as np
import pickle

def randarg(p) :
    cump=p.cumsum()
    x=np.random.random()
    if x>cump[-1]:
        return -1
    k=0
    while x>cump[k] :
        k+=1
    return k
class DataEnv:
    def __init__(self,N,K,
                 fname,
                 episodes_name="train",
                 seed=None,
                 choice_fnct='mnl',
                 static=False,
                 time_budget_0=200.,
                 no_click_mass=1.,
                 min_quality=-3.,
                 max_quality=3.,
                 watch_time=4.,
                 penalty=1.,
                 user_quality_factor=0.,
                 document_quality_factor=1.,
                 update_alpha=.2) :
        self.N=N
        self.N=N
        self.K=K

        self.choice_fnct=choice_fnct
        self.seed=seed

        self.static=static
        self.time_budget_0=time_budget_0
        self.no_click_mass=no_click_mass

        self.user_quality_factor=user_quality_factor
        self.document_quality_factor=document_quality_factor
        self.update_alpha=update_alpha

        self.watch_time=watch_time
        self.penalty=penalty

        with open(fname+"users.npy",'rb') as f :
            self.users=np.load(f)
        with open(fname+"items.npy",'rb') as f :
            self.items=np.load(f)
        with open(fname+episodes_name+"_episodes.pckl",'rb') as f :
            self.episodes=pickle.load(f)

        self.episodes_ids=list(self.episodes.keys()) #list of available episode
        
        self.d_in=self.users.shape[1]
        long = np.linspace(min_quality, 0, int(self.d_in * 0.7))
        short = np.linspace(0, max_quality,
                                self.d_in-int(self.d_in * 0.7))
        self.cluster_means = np.concatenate((long, short))

    def get_params(self) :
        return {'N':self.N,
                'K':self.K,
                'd':self.d_in,
                'tabular':False,
                'user':None,
                'no_click_mass':self.no_click_mass} 

    def sample_user(self,seed=None) :
        if seed is not None :
            np.random.seed(seed)
        self.user_id=np.random.choice(self.episodes_ids)
        self.user=self.users[self.user_id]

        self.rewards={} #dictionnary item_id:reward for knwon ratings
        for item_id,reward in self.episodes[self.user_id] :
            self.rewards[item_id]=reward

        self.time_budget=self.time_budget_0

    def sample_candidates(self) :
        self.candidates_id=np.random.choice(list(self.rewards.keys()),self.N)
        #np.random.shuffle(self.candidates_id)
        self.candidates=self.items[self.candidates_id]
    
    def choice(self,slate) :
        """
        Arg : 
           action: [K] int array, candidates_id
        Returns:
          int, position in the slate
        """
        if self.choice_fnct == 'max' :
            rewards=[self.rewards[i] for i in slate]
            return np.argmax(rewards)

        elif self.choice_fnct == 'random' :
            return np.random.randint(0,self.K)
        
        elif self.choice_fnct == 'mnl' :
            rewards=np.array([self.rewards[i] for i in slate])
            probas=np.exp(rewards)/(np.exp(rewards).sum()+np.exp(self.no_click_mass))
            return randarg(probas)
        
    def update_state(self,itemid) :
        """
        Args :
           item: int, itemid
        """
        if itemid is None :
            self.time_budget=max(0,self.time_budget-self.penalty)
        else :
            item=self.items[itemid]
            expected_utility=np.exp(self.user@item)/(np.exp(self.user@item)+np.exp(self.no_click_mass))

            #update user
            if not self.static :
                alpha = -0.3 * np.abs(self.user) + 0.3
                target = item - self.user 
                update = alpha * item * target

                positive_update_prob =((self.user+1)/2)@item
                flip = np.random.rand(1)
                if flip < positive_update_prob:
                    self.user += update
                else:
                    self.user -= update
                self.user = np.clip(self.user, -1.0, 1.0)

            #update time_budget
            doc_quality=item@self.cluster_means + 0.1*np.random.randn()
            received_utility=(self.user_quality_factor * expected_utility) + (self.document_quality_factor * doc_quality)

            watch_time=min(self.time_budget,self.watch_time)
            self.time_budget-=watch_time
            self.time_budget+=self.update_alpha*watch_time*received_utility
            
            self.time_budget=max(0,self.time_budget)

    def reset(self) :
        self.sample_user()
        self.sample_candidates()
        return self.user, self.candidates

    def step(self,action) :
        """
        Args:
           action: [N] array, indices of slate
        """

        #Choix d'item
        slate=self.candidates_id[action]
        c=self.choice(slate)

        #compute reward
        if c>= 0 :
            reward = self.rewards[slate[c]]
        else :
            reward = 0

        #update time budget
        if c==-1 :
            self.update_state(None)
        else :
            self.update_state(slate[c])

        #generate new candidate
        self.sample_candidates()

        #compute done
        done = self.time_budget == 0
        return self.candidates, c, reward, done