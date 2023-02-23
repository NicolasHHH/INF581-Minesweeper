import pygame

from game import MineSweeper
from renderer import Render


class Play:
    def __init__(self, rule="win7", auto_flag=False, auto_play=False):
        self.width = 9
        self.height = 9
        self.bombs = 10
        self.env = MineSweeper(self.width, self.height, self.bombs, rule=rule)
        self.renderer = Render(self.env.state)
        self.renderer.state = self.env.state

    def do_step(self, i, j):
        i = int(i / 30)
        j = int(j / 30)
        next_state, terminal, reward = self.env.choose(i, j, auto_flag=auto_flag, auto_play=auto_play)
        self.renderer.state = self.env.state
        self.renderer.draw()
        return next_state, terminal, reward


if __name__ == "__main__":
    rule = "winxp" # "default" "winxp"
    auto_flag = False  # mark MUST-BE mines in red
    auto_play = False  # auto play whenever there's a mine-free click

    play = Play(rule, auto_flag, auto_play)
    play.renderer.draw()
    print(play.env.grid)
    while True:
        events = play.renderer.bugfix()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                y, x = pygame.mouse.get_pos()
                _, terminal, reward = play.do_step(x, y)
                if terminal:
                    if reward == -1:
                        print("LOSS")
                    else:
                        print("WIN")
                    play.env.reset()
                    play.renderer.state = play.env.state
                    play.renderer.draw()
                print(play.env.state)
