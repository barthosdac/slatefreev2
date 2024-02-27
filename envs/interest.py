import numpy as np
def randarg(p) :
    cump=p.cumsum()
    x=np.random.random()
    if x>cump[-1]:
        return -1
    k=0
    while x>cump[k] :
        k+=1
    return k
class InterestEnv:
    def __init__(self,N,K,d_in,
                 mono_utilisateur=False,
                 static=False,
                 seed=None,
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
        self.K=K
        self.d_in=d_in

        self.static=static
        self.mono_utilisateur=mono_utilisateur
        self.time_budget_0=time_budget_0
        self.no_click_mass=no_click_mass

        self.user_quality_factor=user_quality_factor
        self.document_quality_factor=document_quality_factor
        self.update_alpha=update_alpha

        self.watch_time=watch_time
        self.penalty=penalty

        trashy = np.linspace(min_quality, 0, int(d_in * 0.7))
        nutritious = np.linspace(0, max_quality,
                                d_in-int(d_in * 0.7))
        self.cluster_means = np.concatenate((trashy, nutritious))

        self.candidates=None
        self.time_budget=None
        if mono_utilisateur :
            self.sample_user(seed)

    def get_params(self) :
        return {'N':self.N,
                'K':self.K,
                'd':self.d_in,
                'tabular':False,
                'no_click_mass':self.no_click_mass} 
    
    def sample_user(self,seed=None) :
        if seed is not None :
            np.random.seed(seed)
        self.user=2*np.random.rand(self.d_in)-1
        self.time_budget=self.time_budget_0

    def sample_candidates(self) :
        self.candidates=np.zeros((self.N,self.d_in))
        for i in range(self.N) :
            self.candidates[i,np.random.randint(0,self.d_in)]=1
    
    def choice(self,slate) :
        """
        Arg : 
           [K,d_in]
        """
        scores=slate@self.user
        p=np.exp(scores)/(np.exp(scores).sum()+np.exp(self.no_click_mass))
        return randarg(p)
    
    def reset(self) :
        if self.mono_utilisateur :
            self.time_budget=self.time_budget_0
        else :
            self.sample_user()

        self.sample_candidates()
        return self.user,self.candidates
    
    def update_state(self,item) :
        if item is None :
            self.time_budget=max(0,self.time_budget-self.penalty)
        else :
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



    def step(self,action) :
        """
        Args:
           action: [N] array, indices of slate
        """

        #Choix d'item
        slate=self.candidates[action]
        c=self.choice(slate)

        #compute reward
        if c==-1 :
            reward = 0
        else :
            reward = min(self.time_budget,self.watch_time)
        #update time budget
        if c==-1 :
            self.update_state(None)
        else :
            self.update_state(slate[c])

        #generate new candidate
        self.sample_candidates()

        #compute done
        done = self.time_budget==0

        return self.candidates, c, reward, done