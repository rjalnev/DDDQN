from dddqn import DDDQNAgent
#from utils import install_roms_in_folder

GAMES = ['SpaceInvaders-Nes', 'Joust-Nes', 'SuperMarioBros-Nes', 'MsPacMan-Nes']
COMBOS = [ [['LEFT'], ['RIGHT'], ['A'], ['LEFT', 'A'], ['RIGHT', 'A']],
           [['LEFT'], ['RIGHT'], ['UP'], ['DOWN']] ]

if __name__ == '__main__':

    #install roms
    #install_roms_in_folder('roms/')

    #create agent
    dddqn = DDDQNAgent(GAMES[2], COMBOS[0], epsilon_decay=0.99999, batch_size=32)
    dddqn.q_eval.summary()
    dddqn.q_target.summary()
    
    #train agent
    dddqn.run(num_episodes=10000, checkpoint=True, cp_interval=100, cp_render=True)
    
    #load model
    dddqn.load('models', 'DDDQN_8800_SpaceInvaders_QEval.h5', 'DDDQN_8800_SpaceInvaders_QTarget.h5')
    
    #play game
    dddqn.play_episode(render=True, render_and_save=True, otype='GIF')