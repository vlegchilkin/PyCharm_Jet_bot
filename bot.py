# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa F401
from collections.abc import Callable

import numpy as np

from vendeeglobe import (
    Checkpoint,
    Heading,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface, goto, wind_force

class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "PyCharm JetBot"  # This is your team name
        self.step = 0
        self.course = []
        self.zones = dict()
        def zone(checkpoint, jiggle=45.0):
            self.zones[len(self.course)] = jiggle
            self.course += [checkpoint]
            return checkpoint

        # This is the course that the ship has to follow
        _ = [
            zone(Checkpoint(latitude=46.8, longitude=-4.5, radius=50)),
            zone(Checkpoint(latitude=58.7, longitude=-42.7, radius=50)),
            zone(Checkpoint(latitude=59.7, longitude=-46.7, radius=50)),
            zone(Checkpoint(latitude=60.7, longitude=-50, radius=50)),
            zone(Checkpoint(latitude=73.9, longitude=-76.8, radius=50)),
            zone(Checkpoint(latitude=74.341717, longitude=-93.926240, radius=50)),
            zone(Checkpoint(latitude=73.8, longitude=-113.519828, radius=50)),
            zone(Checkpoint(latitude=75.3, longitude=-122.296479, radius=50)),
            zone(Checkpoint(latitude=74.7, longitude=-126.296479, radius=50)),
            zone(Checkpoint(latitude=72.0, longitude=-130.0, radius=50)),

            zone(Checkpoint(latitude=71.0, longitude=-130.5, radius=50)),
            zone(Checkpoint(latitude=70.7, longitude=-135.8, radius=50)),
            zone(Checkpoint(latitude=71.0, longitude=-144, radius=50)),
            zone(Checkpoint(latitude=71.45, longitude=-159.5, radius=50)),
            zone(Checkpoint(latitude=68.5, longitude=-168.0, radius=50)),
            zone(Checkpoint(latitude=65.99, longitude=-168.3, radius=50)),
            zone(Checkpoint(latitude=64.877243, longitude=-168.620675, radius=50)),

            zone(Checkpoint(latitude=63.568421, longitude=-167.4, radius=50)),
            zone(Checkpoint(latitude=62.568421, longitude=-167.948765, radius=50)),
            zone(Checkpoint(latitude=60.4, longitude=-171, radius=50)),
            zone(Checkpoint(latitude=51.839381, longitude=-179.35, radius=50)),
            zone(Checkpoint(latitude=29, longitude=179.9, radius=50.0)),
            zone(Checkpoint(latitude=16, longitude=179.9, radius=50.0)),  # check 1
            zone(Checkpoint(latitude=14.2, longitude=169, radius=50.0)),
            zone(Checkpoint(latitude=12.2, longitude=162, radius=50.0)),
            zone(Checkpoint(latitude=9.2, longitude=152, radius=50.0)),
            zone(Checkpoint(latitude=8, longitude=149.5, radius=50.0)),
            zone(Checkpoint(latitude=6.7, longitude=146, radius=50.0)),
            # entry
            zone(Checkpoint(latitude=1, longitude=128.8, radius=50.0)),
            zone(Checkpoint(latitude=-2.5, longitude=128.8, radius=50.0), 10),
            zone(Checkpoint(latitude=-2.8, longitude=126.25, radius=50.0)),
            zone(Checkpoint(latitude=-2.9, longitude=125.25, radius=50.0)),
            zone(Checkpoint(latitude=-7.8, longitude=125.322399, radius=50.0)),
            zone(Checkpoint(latitude=-8.8, longitude=125.322399, radius=50.0), 0),
            zone(Checkpoint(latitude=-8.8, longitude=124.8, radius=50.0), 0),
            zone(Checkpoint(latitude=-10, longitude=121.346855, radius=50.0)),
            zone(Checkpoint(latitude=-11, longitude=120, radius=50.0)),
            zone(Checkpoint(latitude=-9.5, longitude=106, radius=50.0)),
            # Checkpoint(latitude=-9.2, longitude=116.866910, radius=50.0),
            zone(Checkpoint(latitude=-5.5, longitude=80, radius=50.0)),  # check 2
            zone(Checkpoint(latitude=-1, longitude=73, radius=50.0)),
            zone(Checkpoint(latitude=11.5, longitude=52.0, radius=50.0)),
            zone(Checkpoint(latitude=13, longitude=51.0, radius=50.0)),
            #b-channel
            zone(Checkpoint(latitude=11.8, longitude=44.3, radius=50.0)),
            zone(Checkpoint(latitude=12.2, longitude=43.45, radius=50.0)),
            zone(Checkpoint(latitude=14.2, longitude=42.3, radius=50.0), 0),

            zone(Checkpoint(latitude=26.9, longitude=34.8, radius=50.0)),
            zone(Checkpoint(latitude=27.6, longitude=34.0, radius=50.0), 10),
            zone(Checkpoint(latitude=28.2, longitude=33.3, radius=50.0), 0),

            zone(Checkpoint(latitude=29.9, longitude=32.3, radius=50.0), 0),
            zone(Checkpoint(latitude=32.5, longitude=32.4, radius=50.0), 0),

            zone(Checkpoint(latitude=34.5, longitude=23, radius=50.0)),
            zone(Checkpoint(latitude=36.3, longitude=14.7, radius=50.0)),
            zone(Checkpoint(latitude=37.2, longitude=12.3, radius=50.0)),

            zone(Checkpoint(latitude=38.5, longitude=10, radius=50.0)),
            zone(Checkpoint(latitude=38, longitude=4, radius=50.0)),
            zone(Checkpoint(latitude=36.2, longitude=-2, radius=50.0)),
            zone(Checkpoint(latitude=36, longitude=-4.8, radius=50.0)),

            zone(Checkpoint(latitude=36, longitude=-6.25, radius=50.0), 0),
            zone(Checkpoint(latitude=36.7, longitude=-10.5, radius=50.0)),
            zone(Checkpoint(latitude=43.797109, longitude=-10.5, radius=50.0)),
            zone(Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=5,
            )),
        ]

    def run(
            self,
            t: float,
            dt: float,
            longitude: float,
            latitude: float,
            heading: float,
            speed: float,
            vector: np.ndarray,
            forecast: Callable,
            world_map: Callable,
    ) -> Instructions:
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()

        current_position_forecast = forecast(
            latitudes=latitude, longitudes=longitude, times=0
        )
        # current_position_terrain = world_map(latitudes=latitude, longitudes=longitude)
        # ===========================================================
        def heading(cp: Checkpoint, index: int) -> Heading:
            angle = goto(Location(longitude, latitude), Location(longitude=cp.longitude, latitude=cp.latitude))
            h = angle * np.pi / 180.0
            ship_vector = np.array([np.cos(h), np.sin(h)])
            wf = wind_force(ship_vector, np.array(current_position_forecast))
            dist = np.linalg.norm(wf)
            if dist < 8:
                self.step = -1 if self.step == 1 else 1
                jiggle = self.zones[index]
                angle += self.step * jiggle

            return Heading(angle)

        # Go through all checkpoints and find the next one to reach
        for idx, ch in enumerate(self.course):
            if ch.reached:
                continue

            # Compute the distance to the checkpoint
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )
            # Consider slowing down if the checkpoint is close
            jump = dt * np.linalg.norm(speed)
            if dist < 2.0 * ch.radius + jump:
                instructions.sail = min(ch.radius / jump, 1)
            else:
                instructions.sail = 1.0
            # Check if the checkpoint has been reached
            if dist < ch.radius:
                ch.reached = True
            else:
                instructions.heading = heading(ch, idx)
                break

        return instructions
