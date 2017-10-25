import os
import tkinter as tk
import random
import numpy as np

from ddpg import ddpg_learner

WIDTH = 800
HEIGHT = 800
ACTION_DIM = 2
UNIT_STATE_DIM = 11


class Unit:
    def __init__(self, id, x, y, radius, v, team, color, range, cd_time, damage, health):
        self.x = x
        self.y = y
        self.radius = radius
        self.range = range
        self.cd_time = cd_time
        self.damage = damage
        self.v = v
        self.team = team
        self.color = color
        self.id = id
        self.health = health
        self.cd = 0
        self.max_health = health
        self.killed = False

    def distance(self, unit):
        dist = np.hypot(unit.x - self.x, unit.y - self.y)
        return dist


class QuickAgent:
    def __init__(self, unit):
        self.unit = unit
        self.last_target = None
        self.noise = ddpg_learner.OUNoise(2)

    def move(self, world):
        self.unit.cd = max(0, self.unit.cd - 1)
        if self.unit.killed:
            return
        closest, closest_dist = self.find_closest(world)
        if closest is None:
            return
        self.make_coord_move(closest, closest_dist, world)
        self.correct_bounds()

    def shoot(self, world):
        if self.unit.killed:
            return
        closest, closest_dist = self.find_closest(world)
        if self.unit.range >= closest_dist and self.unit.cd == 0:
            h = closest.unit.health
            new_health = max(0, h - self.unit.damage)
            closest.unit.health = new_health
            self.unit.cd = self.unit.cd_time
            self.last_target = closest

    def make_coord_move(self, closest, closest_dist, world):
        coord = np.array([closest.unit.x - self.unit.x, closest.unit.y - self.unit.y])
        norm = np.linalg.norm(coord)
        vec = (coord / norm) + self.noise.noise()
        norm = np.linalg.norm(vec)
        vec = (vec / norm) * self.unit.v

        self.unit.x += vec[0]
        self.unit.y += vec[1]
        closest, closest_dist = self.find_closest_with_team(world)
        new_dist = np.hypot(self.unit.x - closest.unit.x, self.unit.y - closest.unit.y)
        if new_dist < self.unit.radius + closest.unit.radius + 1:
            self.unit.x -= vec[0]
            self.unit.y -= vec[1]

    def find_closest(self, world):
        closest = None
        closest_dist = 1000000
        for agent in world.agents:
            other = agent.unit
            u = self.unit
            if u.team != other.team and other.health > 0:
                dist = u.distance(other)
                if dist < closest_dist:
                    closest_dist = dist
                    closest = agent
        return closest, closest_dist

    def find_closest_with_team(self, world):
        closest = None
        closest_dist = 1000000
        for agent in world.agents:
            other = agent.unit
            u = self.unit
            if self.unit.id != other.id and other.health > 0:
                dist = u.distance(other)
                if dist < closest_dist:
                    closest_dist = dist
                    closest = agent
        return closest, closest_dist

    def draw(self, canvas):
        u = self.unit
        if u.health <= 0:
            return
        canvas.create_oval(u.x - u.radius, u.y - u.radius, u.x + u.radius, u.y + u.radius,
                           outline=u.color, fill=u.color)
        width = int(u.radius * 2 / u.max_health * u.health)
        canvas.create_rectangle(u.x - u.radius, u.y + u.radius + 2, u.x + u.radius, u.y + u.radius + 4,
                                outline='black', fill='white')
        canvas.create_rectangle(u.x - u.radius, u.y + u.radius + 2, u.x - u.radius + width, u.y + u.radius + 4,
                                outline='black', fill='black')
        if u.cd_time - u.cd < 3 and self.last_target is not None:
            canvas.create_line(u.x, u.y, self.last_target.unit.x, self.last_target.unit.y,
                               fill='black')

    def before_move(self, world):
        pass

    def after_move(self, world):
        pass

    def correct_bounds(self):
        u = self.unit
        if u.x - u.radius < 0:
            u.x = u.radius
        if u.x + u.radius > WIDTH:
            u.x = WIDTH - u.radius
        if u.y - u.radius < 0:
            u.y = u.radius
        if u.y + u.radius >= HEIGHT:
            u.y = HEIGHT - u.radius


class DDPGAgent(QuickAgent):
    def __init__(self, unit, actor):
        super().__init__(unit)
        self.last_state = None
        self.actor = actor
        self.picked_action = None
        self.health_diff = 0
        self.noise = ddpg_learner.OUNoise(2)

    def make_coord_move(self, closest, closest_dist, world):
        suggested_action = self.actor.get_action(self.last_state)
        if world.step % 300 == 0:
            print(suggested_action)
        a = suggested_action + self.noise.noise()  # * 0.5
        vx = a[0]
        vy = a[1]
        if np.hypot(vx, vy) > 1.0:
            a /= np.linalg.norm(a)
        self.unit.x += a[0] * self.unit.v
        self.unit.y += a[1] * self.unit.v
        closest, closest_dist = self.find_closest_with_team(world)
        new_dist = np.hypot(self.unit.x - closest.unit.x, self.unit.y - closest.unit.y)
        if new_dist < self.unit.radius + closest.unit.radius + 1:
            self.unit.x -= a[0] * self.unit.v
            self.unit.y -= a[1] * self.unit.v
        self.picked_action = a

    def before_move(self, world):
        self.last_state = self.get_state(world)
        self.health_diff = self.compute_health_diff(world)

    def after_move(self, world):
        cur_state = self.get_state(world)
        cur_health = self.compute_health_diff(world)
        reward = cur_health - self.health_diff
        if world.finished:
            if self.unit.team in world.teams:
                reward = 100
            elif len(world.teams) == 1:
                reward = -100
        #reward /= 100
        self.actor.add(self.last_state, self.picked_action, reward, cur_state, world.finished)

    def compute_health_diff(self, world):
        hd = 0
        for a in world.agents:
            if a.unit.team == self.unit.team:
                hd += a.unit.health
            else:
                hd -= a.unit.health
        return hd

    def get_state(self, world):
        cur = self.get_unit_state(self.unit)
        for agent in world.agents:
            if self.unit.id != agent.unit.id:
                cur += self.get_unit_state(agent.unit)
        return np.array(cur) / 100.0

    @staticmethod
    def get_unit_state(unit):
        s = [unit.x - WIDTH / 2,
             unit.y - HEIGHT / 2,
             unit.radius,
             unit.range,
             unit.cd_time,
             unit.damage,
             unit.v,
             unit.team,
             unit.health,
             unit.max_health,
             unit.cd]
        return s


class App(object):
    def __init__(self, drawable, world):
        self.drawable = drawable
        if drawable:
            self.master = tk.Tk()
            self.canvas = self.canvas = tk.Canvas(self.master, width=WIDTH, height=HEIGHT)
            self.canvas.pack()
        self.world = world

    def start(self):
        self.world.reset()
        if self.drawable:
            self.master.after(0, self.animate)
            self.master.mainloop()
        else:
            self.animate()

    def animate(self):
        while True:
            self.world.tick()
            if self.drawable:
                # time.sleep(0.01)
                self.world.draw(self.canvas)
                self.canvas.update()


class World:
    def __init__(self, actor):
        self.agents = []
        self.id = 0
        self.finished = False
        self.actor = actor
        self.step = 0
        self.cum_loss = 0
        self.loss_decay = 0.999
        self.teams = set()
        self.list_wins = []

    def tick(self):
        self.step += 1
        self.teams = set()
        for a in self.agents:
            a.before_move(self)
        for a in self.agents:
            a.move(self)
        for a in self.agents:
            a.shoot(self)
        for a in self.agents:
            if a.unit.health > 0:
                self.teams.add(a.unit.team)
            if a.unit.health <= 0:
                a.unit.killed = True
        if len(self.teams) < 2:
            self.finished = True
            if len(self.teams) == 1:
                t = next(iter(self.teams))
                self.list_wins.append(t)
                if len(self.list_wins) > 100:
                    self.list_wins = self.list_wins[1:]
                cnt_wins = 0
                for x in self.list_wins:
                    if x == 1:
                        cnt_wins += 1
                print("Win rate: %s" % (cnt_wins / len(self.list_wins)))
        for a in self.agents:
            a.after_move(self)
        if self.finished:
            self.reset()
            self.finished = False
        self.actor.make_train()
        if self.cum_loss == 0:
            self.cum_loss = self.actor.last_loss
        else:
            self.cum_loss = self.cum_loss * self.loss_decay + self.actor.last_loss * (1 - self.loss_decay)
            if self.step % 100 == 0:
                print("Loss: %s. At step: %s" % (self.cum_loss, self.step))

    def reset(self):
        self.agents = []
        for i in range(5):
            while len(self.agents) < 12:
                cur_x = random.randrange(50, WIDTH // 2 - 150)
                cur_y = random.randrange(50, HEIGHT - 50)
                quick_melee = QuickAgent(self.create_melee_unit(cur_x, cur_y, -1))
                agent_melee = DDPGAgent(self.create_melee_unit(WIDTH - cur_x, HEIGHT - cur_y, 1), self.actor)
                suit_1 = self.is_suitable(quick_melee)
                suit_2 = self.is_suitable(agent_melee)
                if suit_1 and suit_2:
                    self.agents.append(quick_melee)
                    self.agents.append(agent_melee)

    def is_suitable(self, candidate):
        closest = None
        closest_dist = 1000000
        for agent in self.agents:
            other = agent.unit
            dist = candidate.unit.distance(other)
            if dist < closest_dist:
                closest_dist = dist
                closest = agent
        if closest is None:
            return True
        if closest_dist <= candidate.unit.radius + closest.unit.radius:
            return False
        return True

    def create_melee_unit(self, x, y, team):
        if team > 0:
            color = "red"
        else:
            color = "green"
        u = Unit(self.id, x, y, 10, 10, team, color, 150, 15, 15, 100)
        self.id += 1
        return u

    def draw(self, canvas):
        canvas.delete("all")
        for a in self.agents:
            a.draw(canvas)
        canvas.update()


np.set_printoptions(suppress=True)

if not os.path.exists("model_data"):
    os.mkdir("model_data")

runner = ddpg_learner.create_runner("model_data/model", UNIT_STATE_DIM * 12, ACTION_DIM,
                                    batch_size=64, buffer_size=1400000)

print("Warming up for 25000 ticks")

my_world = World(runner)
my_world.reset()
app = App(False, my_world)
app.start()
