import math
from typing import Optional, Union

import numpy as np

import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle


try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[box2d]`"
    ) from e


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale

FPS = 50  # Frames per second
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 100 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        if tile.goal:
            tile.color[:] = self.env.goal_color
        else:
            tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)


class CarRiverCrossing(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gymnasium/envs/box2d/car_river_crossing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: breaking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer left
    - 2: steer right
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go
    outside the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.

    ## Arguments
    `lap_complete_percent` dictates the percentage of tiles that must be visited by
    the agent before a lap is considered complete.

    Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    In this scenario, the background and track colours are different on every reset.

    Passing `continuous=False` converts the environment to use discrete action space.
    The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ## Reset Arguments
    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.
    Example usage:
    ```python
    import gymnasium as gym
    env = gym.make("CarRacing-v1", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    env.reset()

    # reset with colour scheme change
    env.reset(options={"randomize": True})

    # reset with no colour scheme change
    env.reset(options={"randomize": False})
    ```

    ## Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        continuous: bool = False,
        play_field_area: int = 300,
        zoom: float = 2.0,
        river_force: float = 1.0,
        river_drag_prob: float = 0.4,
        river_brake_force: float = 0.4
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            continuous,
        )

        self.zoom = zoom  # Camera zoom
        self.play_field = play_field_area / SCALE  # Game over boundary
        self.river_force = river_force
        self.river_drag_prob = river_drag_prob
        self.river_brake_force = river_brake_force

        real_track_width = (TRACK_WIDTH * SCALE)/2
        adjusted_track = play_field_area / real_track_width
        new_real_track_width = play_field_area / math.ceil(adjusted_track)
        self.adjusted_track_width = (new_real_track_width*2)/SCALE
        print('TRACK_WIDTH', TRACK_WIDTH, self.adjusted_track_width)

        self.grass_dim = self.play_field / 20.0
        self.max_shape_dim = (
                max(self.grass_dim, self.adjusted_track_width, TRACK_DETAIL_STEP) * math.sqrt(2) * self.zoom * SCALE
        )

        self.continuous = continuous
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        # default colours
        self.road_color = np.array([102, 102, 102])
        self.bg_color = np.array([102, 102, 204])
        self.grass_color = np.array([102, 102, 230])
        self.mud_color = np.array([0, 130, 0])
        self.goal_color = np.array([230, 102, 102])

    def _create_track(self):
        self.road = []

        # draw left
        y1 = -self.play_field
        x1 = -self.play_field+self.adjusted_track_width
        while True:
            self._add_tile(x1, y1, False)
            y1 += self.adjusted_track_width
            if y1 >= (self.play_field - self.adjusted_track_width):
                break

        # draw top
        y1 = self.play_field - (2*self.adjusted_track_width)
        x1 = -self.play_field + self.adjusted_track_width
        while True:
            self._add_tile(x1, y1, False)
            x1 += self.adjusted_track_width
            if x1 >= (self.play_field - self.adjusted_track_width):
                break

        # draw right
        y1 = self.play_field - (2*self.adjusted_track_width)
        x1 = self.play_field - (2*self.adjusted_track_width)
        while True:
            if y1 <= -self.play_field:
                self._add_tile(x1, y1, True)
            else:
                self._add_tile(x1, y1, False)
            y1 -= self.adjusted_track_width
            if y1 <= -self.play_field - self.adjusted_track_width:
                break
        return True

    def _add_tile(self, x1, y1, goal):
        road1_l = (
            x1,
            y1,
        )
        road1_r = (
            x1 + self.adjusted_track_width,
            y1,
        )
        road2_l = (
            x1,
            y1 + self.adjusted_track_width,
        )
        road2_r = (
            x1 + self.adjusted_track_width,
            y1 + self.adjusted_track_width,
        )

        vertices = [road1_l, road1_r, road2_r, road2_l]
        self.fd_tile.shape.vertices = vertices
        t = self.world.CreateStaticBody(fixtures=self.fd_tile)
        t.userData = t
        t.goal = goal
        if goal:
            t.color = self.goal_color
        else:
            t.color = self.road_color
        t.road_visited = False
        t.road_friction = 1.0
        t.idx = len(self.road)
        t.fixtures[0].sensor = True
        self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
        self.road.append(t)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        reward: float = 0.0
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = reward
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        # self.car = Car(self.world, *self.track[0][1:4])
        x = (self.road_poly[0][0][0][0] + self.road_poly[0][0][2][0]) / 2
        y = (self.road_poly[0][0][0][1] + self.road_poly[0][0][2][1]) / 2
        self.car = Car(self.world, 0, x, y)

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None
        if action is not None:

            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        step_reward = -0.1
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            # ignoring that
            self.car.fuel_spent = 0.0

            touched_goal = any(t.goal for w in self.car.wheels for t in w.tiles)
            if touched_goal:
                terminated = True
                step_reward = 0

            self.reward += step_reward

            # out fo bounds - reset
            x, y = self.car.hull.position
            if abs(x) > self.play_field or abs(y) > self.play_field:
                truncated = True
                # reset but keep reward
                self.reset(reward=self.reward)

            # apply river force
            # it has a change to be dragged
            # if any wheel touched the water it will be dragged
            touched_tiles = sum(len(w.tiles) > 0 for w in self.car.wheels)
            if touched_tiles < 4:
                if np.random.rand() <= self.river_drag_prob:
                    self.car.brake(self.river_brake_force)
                    # 1 touched mud on left
                    # 2 touched mud on top
                    # 3 touched mud on right
                    touched_mud = 0
                    for w in self.car.wheels:
                        x, y = w.position
                        if x < -self.play_field + self.adjusted_track_width:
                            touched_mud = 1
                        elif x > self.play_field - self.adjusted_track_width:
                            touched_mud = 3

                    self.car.hull.position = self.recalculate_position(self.car.hull.position, touched_mud)
                    for w in self.car.wheels:
                        w.position = self.recalculate_position(w.position, touched_mud)

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        if self.render_mode == "human":
            self.render()
        info = {'position': self.car.hull.position}
        return self.state, step_reward, terminated, truncated, info

    def recalculate_position(self, position, touched_mud):
        x, y = position
        if touched_mud == 1:
            return x + self.river_force, y
        elif touched_mud == 3:
            return x - self.river_force, y
        else:
            return x, y - self.river_force


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + self.zoom * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
             self.surf,
             zoom,
             trans,
             angle,
             mode not in ["state_pixels_list", "state_pixels"],
         )

        #DESENHAR


        #gfxdraw.pixel(self.surf, -25, -41, dot_color)

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        # hiding rewards when it is not human
        if mode == "human":
            self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = self.play_field
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw river patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (self.grass_dim * x + self.grass_dim, self.grass_dim * y + 0),
                        (self.grass_dim * x + 0, self.grass_dim * y + 0),
                        (self.grass_dim * x + 0, self.grass_dim * y + self.grass_dim),
                        (self.grass_dim * x + self.grass_dim, self.grass_dim * y + self.grass_dim),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw mud
        mud_area_left = [
            (-self.play_field + self.adjusted_track_width, self.play_field),
            (-self.play_field + self.adjusted_track_width, -self.play_field),
            (-self.play_field, -self.play_field),
            (-self.play_field, self.play_field),
        ]
        mud_area_top = [
            (self.play_field, self.play_field),
            (self.play_field, self.play_field - self.adjusted_track_width),
            (-self.play_field, self.play_field - self.adjusted_track_width),
            (-self.play_field, self.play_field),
        ]
        mud_area_right = [
            (self.play_field, self.play_field),
            (self.play_field, -self.play_field),
            (self.play_field - self.adjusted_track_width, -self.play_field),
            (self.play_field - self.adjusted_track_width, self.play_field),
        ]
        self._draw_colored_polygon(
            self.surf, mud_area_left, self.mud_color, zoom, translation, angle, clip=False
        )
        self._draw_colored_polygon(
            self.surf, mud_area_top, self.mud_color, zoom, translation, angle, clip=False
        )
        self._draw_colored_polygon(
            self.surf, mud_area_right, self.mud_color, zoom, translation, angle, clip=False
        )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-self.max_shape_dim <= coord[0] <= WINDOW_W + self.max_shape_dim)
            and (-self.max_shape_dim <= coord[1] <= WINDOW_H + self.max_shape_dim)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    # env = CarRiverCrossing(continuous=True, render_mode="human", play_field_area=300, zoom=1)
    env = CarRiverCrossing(continuous=True, render_mode="human", play_field_area=300, zoom=1.0)

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or restart or quit:
                break
    env.close()