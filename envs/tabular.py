import numpy as np
class TabularEnv:
    def __init__(self,N,K,
                 size_set=3,
                 user_type=1,
                 time_budget_0=10,

                 mono_utilisateur=False,
                 seed=None) :
        self.N=N
        self.K=K

        self.size_set=size_set
        self.user_type=user_type

        self.mono_utilisateur=mono_utilisateur
        self.seed=seed

        self.time_budget_0=time_budget_0

        self.sample_corpus(seed)
        if mono_utilisateur :
            self.sample_user(seed)

    def get_params(self) :
        return {'N':self.N,
                'K':self.K,
                'd':None,
                'tabular':True,
                'no_click_mass':None} 
    
    def sample_user(self,seed=None) :
        if seed is not None :
            np.random.seed(seed)

        self.time_budget=self.time_budget_0
        self.alpha=0.9

        args=np.arange(self.N)
        np.random.shuffle(args)
        self.set=args[:self.size_set]


    def sample_corpus(self,seed=None) :
        if seed is not None :
            np.random.seed(seed)
        self.rewards=5*np.random.random(self.N)

    def reset(self) :
        if self.mono_utilisateur :
            self.time_budget=self.time_budget_0
        else :
            self.sample_user()
            
        return 
    def choice(self,slate) :
        """
        Args :
          slate: [K] int array, idx of items proposed
        Returns :
          int: idx of chosen item
        """
        if self.user_type==1 :
            if np.random.random()<self.alpha :
                return np.random.choice(slate)
            else :
                return np.random.randint(0,self.N)
        elif self.user_type==2 :
            slate=[k for k in slate if k not in self.set]
            if len(slate)>0 and np.random.random()<self.alpha :
                return np.random.choice(slate)
            else :
                return np.random.randint(0,self.N)
        elif self.user_type==3 :
            for i in self.set :
                if i in slate :
                    return np.random.choice(slate)
            return np.random.randint(0,self.N)
    def step(self,slate) :
        """
        Args:
           action: [K] int array, indices of items in slate
        """

        #Choix d'item
        c=self.choice(slate)

        #compute reward
        reward = self.rewards[c]

        #update time budget
        self.time_budget-=1

        #compute done
        done = self.time_budget==0

        return None, c, reward, done