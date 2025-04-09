from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        # print("It is ", other_per_controlled)
        # 这里是创建控制车
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"] + 1,
            )
            # print("before is ", type(vehicle))
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            # print("after is", type(vehicle))
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        # 这里是创建其他 IDM 车
        for _ in range(others):
            vehicle = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"]
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)


            

############################################################################################################################
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        all_rewards = self._rewards(action)
        # print(all_rewards)
        return_ls = []
        
        for item in all_rewards: # 对

            reward = sum(
                self.config.get(name, 0) * reward for name, reward in item.items()
            )
            # print("thi reward is :", reward)
            if self.config["normalize_reward"]:
                reward = utils.lmap( # 归一化 [collision_reward, high_speed_reward + right_lane_reward]
                    reward,
                    [
                        self.config["collision_reward"],
                        self.config["high_speed_reward"] + self.config["right_lane_reward"],
                    ],
                    [0, 1],
                )
            # print(" it is ", item["on_road_reward"])
            reward *= item["on_road_reward"] #  这里应该是防止车开出去
            # print("this2 reward is :", reward)
            return_ls.append(reward)
        return return_ls




        rewards = self._rewards(action)[0]
        # print(tmp_reward)
        # rewards = tmp_reward
        # print(action)
        # print(rewards.items())
        

        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward
############################################################################################################################



##############################################################################################################################
    def _rewards(self, action: Action) -> dict[str, float]:
        
        return_ls = [] # 返回的是一个 原本的单个车 的 返回的 dict 所组成的 所有控制车 的 dict 的 list
        for tmp_vehicle in self.controlled_vehicles:
        # tmp_vehicle = self.controlled_vehicles[0]
        # assert tmp_vehicle == self.vehicle, "vehicle diff...."

            neighbours = self.road.network.all_side_lanes(tmp_vehicle.lane_index)
            lane = (
                tmp_vehicle.target_lane_index[2]
                if isinstance(tmp_vehicle, ControlledVehicle)
                else tmp_vehicle.lane_index[2]
            )
            # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
            forward_speed = tmp_vehicle.speed * np.cos(tmp_vehicle.heading)
            scaled_speed = utils.lmap(
                forward_speed, self.config["reward_speed_range"], [0, 1]
            )

            return_ls.append({
                "collision_reward": float(tmp_vehicle.crashed),
                "right_lane_reward": lane / max(len(neighbours) - 1, 1),
                "high_speed_reward": np.clip(scaled_speed, 0, 1),
                "on_road_reward": float(tmp_vehicle.on_road),
            })
        
        return return_ls # 这里的 action 没有使用，其实这里可以加上 动作合理性的判断



        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }
############################################################################################################################


############################################################################################################################
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        """
            这里应该还是每一个车有自己的 done 比较合适

        """

        return_ls = [True if x.crashed == True else False for x in self.controlled_vehicles]

        return return_ls

        


        all_car_crashed = False
        
        if all([x.crashed == True for x in self.controlled_vehicles]):
            all_car_crashed = True
        else:
            all_car_crashed = False

        return (
            all_car_crashed # self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )
############################################################################################################################




    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
