import pygame

class Human:
    def __init__(self, input_length):
        self.input_length = input_length
        self.input_status = [0] * self.input_length
        pass

    def get_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Get the status of the number keys 1-9
        keys = pygame.key.get_pressed()
        for i in range(self.input_length):
            if keys[getattr(pygame, f'K_{i+1}')]:
                self.input_status[i] = 1
            else:
                self.input_status[i] = 0
        
        # action = 0
        # for i, status in enumerate(self.input_status):
        #     action += status * 2**i
        #return action
                
        return self.input_status
    