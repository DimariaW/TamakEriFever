import os
import sys
import random
import json
import copy
import enum
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from handyrl_core.model import BaseModel, Dense

from football.util import *
import football.rulebaseA as rulebaseA
import football.rulebaseB as rulebaseB
import football.rulebaseC as rulebaseC
#import football.rulebaseD as rulebaseD
import football.rulebaseE as rulebaseE
import football.rulebaseF as rulebaseF



OBS_TEMPLATE = {
    "controlled_players": 1,
    "players_raw": [
        {
            "right_team_active": [True, True, True, True, True, True, True, True, True, True, True],
            "right_team_yellow_card": [False, False, False, False, False, False, False, False, False, False, False],
            "left_team_tired_factor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "right_team_roles": [0, 2, 1, 1, 3, 5, 5, 5, 6, 9, 7],
            "left_team": [
                [-1.0110293626785278, -0.0],
                [-0.4266543984413147, -0.19894461333751678],
                [-0.5055146813392639, -0.06459399312734604],
                [-0.5055146813392639, 0.06459297984838486],
                [-0.4266543984413147, 0.19894461333751678],
                [-0.18624374270439148, -0.10739918798208237],
                [-0.270525187253952, -0.0],
                [-0.18624374270439148, 0.10739918798208237],
                [-0.010110294446349144, -0.21961550414562225],
                [-0.05055147036910057, -0.0],
                [-0.010110294446349144, 0.21961753070354462]
            ],
            "ball": [0.0, -0.0, 0.11061639338731766],
            "ball_owned_team": -1,
            "right_team_direction": [
                [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0], [-0.0, 0.0]
            ],
            "left_team_direction": [
                [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0]
            ],
            "left_team_roles": [0, 2, 1, 1, 3, 5, 5, 5, 6, 9, 7],
            "score": [0, 0],
            "left_team_active": [True, True, True, True, True, True, True, True, True, True, True],
            "game_mode": 0,
            "steps_left": 3001,
            "ball_direction": [-0.0, 0.0, 0.006163952872157097],
            "ball_owned_player": -1,
            "right_team": [
                [1.0110293626785278, 0.0],
                [0.4266543984413147, 0.19894461333751678],
                [0.5055146813392639, 0.06459399312734604],
                [0.5055146813392639, -0.06459297984838486],
                [0.4266543984413147, -0.19894461333751678],
                [0.18624374270439148, 0.10739918798208237],
                [0.270525187253952, 0.0],
                [0.18624374270439148, -0.10739918798208237],
                [0.010110294446349144, 0.21961550414562225],
                [-0.0, -0.02032535709440708], [-0.0, 0.02032535709440708]
            ],
            "left_team_yellow_card": [False, False, False, False, False, False, False, False, False, False, False],
            "ball_rotation": [0.0, -0.0, 0.0],
            "right_team_tired_factor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "designated": 6,
            "active": 6,
            "sticky_actions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    ]
}

INFO_TEMPLATE = {
    'half_step': 1500
}


# feature
def feature_from_states(states, info, player):
    # observation list to input tensor

    HISTORY_LENGTH = 8

    obs_history_ = [s[player]['observation']['players_raw'][0] for s in reversed(states[-HISTORY_LENGTH:])]
    obs_history = obs_history_ + [obs_history_[-1]] * (HISTORY_LENGTH - len(obs_history_))
    obs = obs_history[0]

    action_history_ = [s[player]['action'][0] for s in reversed(states[-HISTORY_LENGTH:])]
    action_history = action_history_ + [0] * (HISTORY_LENGTH - len(action_history_ ))

    """
    ・left players (x)
    ・left players (y)
    ・right players (x)
    ・right players (y)
    ・ball (x)
    ・ball (y)
    ・left goal (x)
    ・left goal (y)
    ・right goal (x)
    ・right goal (y)
    ・active (x)
    ・active (y)

    ・left players (x) - right players (x)
    ・left players (y) - right players (y)
    ・left players (x) - ball (x)
    ・left players (y) - ball (y)
    ・left players (x) - goal (x)
    ・left players (y) - goal (y)
    ・left players (x) - active (x)
    ・left players (y) - active (y)

    ・left players direction (x)
    ・left players direction (y)
    ・right players direction (x)
    ・right players direction (y)
    ・left players direction (x) - right players direction (x)
    ・left players direction (y) - right players direction (y)
    """

    # left players
    obs_left_team = np.array(obs['left_team'])
    left_player_x = np.repeat(obs_left_team[:, 0][..., None], 11, axis=1)
    left_player_y = np.repeat(obs_left_team[:, 1][..., None], 11, axis=1)

    # right players
    obs_right_team = np.array(obs['right_team'])
    right_player_x = np.repeat(obs_right_team[:, 0][..., None], 11, axis=1).transpose(1, 0)
    right_player_y = np.repeat(obs_right_team[:, 1][..., None], 11, axis=1).transpose(1, 0)

    # ball
    obs_ball = np.array(obs['ball'])
    ball_x = np.ones((11, 11)) * obs_ball[0]
    ball_y = np.ones((11, 11)) * obs_ball[1]
    ball_z = np.ones((11, 11)) * obs_ball[2]

    # goal
    left_goal, right_goal = [-1, 0], [1, 0]
    left_goal_x = np.ones((11, 11)) * left_goal[0]
    left_goal_y = np.ones((11, 11)) * left_goal[1]
    right_goal_x = np.ones((11, 11)) * right_goal[0]
    right_goal_y = np.ones((11, 11)) * right_goal[1]

    # side line
    side_line_y = [-.42, .42]
    side_line_y_top = np.ones((11, 11)) * side_line_y[0]
    side_line_y_bottom = np.ones((11, 11)) * side_line_y[1]

    # active
    active = np.array(obs['active'])
    active_player_x = np.repeat(obs_left_team[active][0][..., None, None], 11, axis=1).repeat(11, axis=0)
    active_player_y = np.repeat(obs_left_team[active][1][..., None, None], 11, axis=1).repeat(11, axis=0)

    # left players - right players
    left_minus_right_player_x = obs_left_team[:, 0][..., None] - obs_right_team[:, 0]
    left_minus_right_player_y = obs_left_team[:, 1][..., None] - obs_right_team[:, 1]

    # left players - ball
    left_minus_ball_x = (obs_left_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1)
    left_minus_ball_y = (obs_left_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1)

    # left players - right goal
    left_minus_right_goal_x = (obs_left_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1)
    left_minus_right_goal_y = (obs_left_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1)

    # left players - left goal
    left_minus_left_goal_x = (obs_left_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1)
    left_minus_left_goal_y = (obs_left_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1)

    # right players - right goal
    right_minus_right_goal_x = (obs_right_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_right_goal_y = (obs_right_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1).transpose(1, 0)

    # right players - left goal
    right_minus_left_goal_x = (obs_right_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_left_goal_y = (obs_right_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1).transpose(1, 0)

    # left players (x) - active
    left_minus_active_x = (obs_left_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11, axis=1)
    left_minus_active_y = (obs_left_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11, axis=1)

    # right player - ball
    right_minus_ball_x = (obs_right_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_ball_y = (obs_right_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1).transpose(1, 0)

    # right player - active
    right_minus_active_x = (obs_right_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_active_y = (obs_right_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11, axis=1).transpose(1, 0)

    # left player - side line
    left_minus_side_top = np.abs(obs_left_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1)
    left_minus_side_bottom = np.abs(obs_left_team[:, 1][..., None] - side_line_y[1]).repeat(11, axis=1)

    # right player - side line
    right_minus_side_top = np.abs(obs_right_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_side_bottom = np.abs(obs_right_team[:, 1][..., None] - side_line_y[1]).repeat(11, axis=1).transpose(1, 0)

    # left players direction
    obs_left_team_direction = np.array(obs['left_team_direction'])
    left_player_direction_x = np.repeat(obs_left_team_direction[:, 0][..., None], 11, axis=1)
    left_player_direction_y = np.repeat(obs_left_team_direction[:, 1][..., None], 11, axis=1)

    # right players direction
    obs_right_team_direction = np.array(obs['right_team_direction'])
    right_player_direction_x = np.repeat(obs_right_team_direction[:, 0][..., None], 11, axis=1).transpose(1, 0)
    right_player_direction_y = np.repeat(obs_right_team_direction[:, 1][..., None], 11, axis=1).transpose(1, 0)

    # ball direction
    obs_ball_direction = np.array(obs['ball_direction'])
    ball_direction_x = np.ones((11, 11)) * obs_ball_direction[0]
    ball_direction_y = np.ones((11, 11)) * obs_ball_direction[1]
    ball_direction_z = np.ones((11, 11)) * obs_ball_direction[2]

    # left players direction - right players direction
    left_minus_right_player_direction_x = obs_left_team_direction[:, 0][..., None] - obs_right_team_direction[:, 0]
    left_minus_right_player_direction_y = obs_left_team_direction[:, 1][..., None] - obs_right_team_direction[:, 1]

    # left players direction - ball direction
    left_minus_ball_direction_x = (obs_left_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(11, axis=1)
    left_minus_ball_direction_y = (obs_left_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(11, axis=1)

    # right players direction - ball direction
    right_minus_ball_direction_x = (obs_right_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(11, axis=1).transpose(1, 0)
    right_minus_ball_direction_y = (obs_right_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(11, axis=1).transpose(1, 0)

    # ball rotation
    obs_ball_rotation = np.array(obs['ball_rotation'])
    ball_rotation_x = np.ones((11, 11)) * obs_ball_rotation[0]
    ball_rotation_y = np.ones((11, 11)) * obs_ball_rotation[1]
    ball_rotation_z = np.ones((11, 11)) * obs_ball_rotation[2]

    cnn_feature = np.stack([
        left_player_x,
        left_player_y,
        right_player_x,
        right_player_y,
        ball_x,
        ball_y,
        ball_z,
        left_goal_x,
        left_goal_y,
        right_goal_x,
        right_goal_y,
        side_line_y_top,
        side_line_y_bottom,
        active_player_x,
        active_player_y,
        left_minus_right_player_x,
        left_minus_right_player_y,
        left_minus_right_goal_x,
        left_minus_right_goal_y,
        left_minus_left_goal_x,
        left_minus_left_goal_y,
        right_minus_right_goal_x,
        right_minus_right_goal_y,
        right_minus_left_goal_x,
        right_minus_left_goal_y,
        left_minus_side_top,
        left_minus_side_bottom,
        right_minus_side_top,
        right_minus_side_bottom,
        right_minus_ball_x,
        right_minus_ball_y,
        right_minus_active_x,
        right_minus_active_y,
        left_minus_ball_x,
        left_minus_ball_y,
        left_minus_active_x,
        left_minus_active_y,
        ball_direction_x,
        ball_direction_y,
        ball_direction_z,
        left_minus_ball_direction_x,
        left_minus_ball_direction_y,
        right_minus_ball_direction_x,
        right_minus_ball_direction_y,
        left_player_direction_x,
        left_player_direction_y,
        right_player_direction_x,
        right_player_direction_y,
        left_minus_right_player_direction_x,
        left_minus_right_player_direction_y,
        ball_rotation_x,
        ball_rotation_y,
        ball_rotation_z,
    ], axis=0)

    # ball
    BALL_OWEND_1HOT = {-1: [0, 0], 0: [1, 0], 1: [0, 1]}
    ball_owned_team_ = obs['ball_owned_team']
    ball_owned_team = BALL_OWEND_1HOT[ball_owned_team_]  # {-1, 0, 1} None, self, opponent
    PLAYER_1HOT = np.concatenate([np.eye(11), np.zeros((1, 11))])
    ball_owned_player_ = PLAYER_1HOT[obs['ball_owned_player']]  # {-1, N-1}
    if ball_owned_team_ == -1:
        my_ball_owned_player = PLAYER_1HOT[-1]
        op_ball_owned_player = PLAYER_1HOT[-1]
    elif ball_owned_team_ == 0:
        my_ball_owned_player = ball_owned_player_
        op_ball_owned_player = PLAYER_1HOT[-1]
    else:
        my_ball_owned_player = PLAYER_1HOT[-1]
        op_ball_owned_player = ball_owned_player_

    ball_features = np.concatenate([
        obs['ball'],
        obs['ball_direction'],
        obs['ball_rotation']
    ]).astype(np.float32)

    # self team
    left_team_features = np.concatenate([
        [[1] for _ in obs['left_team']],  # left team flag
        obs['left_team'],  # position
        obs['left_team_direction'],
        [[v] for v in obs['left_team_tired_factor']],
        [[v] for v in obs['left_team_yellow_card']],
        [[v] for v in obs['left_team_active']],
        my_ball_owned_player[...,np.newaxis]
    ], axis=1).astype(np.float32)

    left_team_indice = np.arange(0, 11, dtype=np.int32)

    # opponent team
    right_team_features = np.concatenate([
        [[0] for _ in obs['right_team']],  # right team flag
        obs['right_team'],  # position
        obs['right_team_direction'],
        [[v] for v in obs['right_team_tired_factor']],
        [[v] for v in obs['right_team_yellow_card']],
        [[v] for v in obs['right_team_active']],
        op_ball_owned_player[...,np.newaxis]
    ], axis=1).astype(np.float32)

    right_team_indice = np.arange(0, 11, dtype=np.int32)

    # distance information
    def get_distance(xy1, xy2):
        return (((xy1 - xy2) ** 2).sum(axis=-1)) ** 0.5

    def get_line_distance(x1, x2):
        return np.abs(x1 - x2)

    def multi_scale(x, scale):
        return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

    both_team = np.array(obs['left_team'] + obs['right_team'], dtype=np.float32)
    ball = np.array([obs['ball'][:2]], dtype=np.float32)
    goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
    goal_line_x = np.array([-1, 1], dtype=np.float32)
    side_line_y = np.array([-.42, .42], dtype=np.float32)

    # ball <-> goal, goal line, side line distance
    b2g_distance = get_distance(ball, goal)
    b2gl_distance = get_line_distance(ball[0][0], goal_line_x)
    b2sl_distance = get_line_distance(ball[0][1], side_line_y)
    b2o_distance = np.concatenate([
        b2g_distance, b2gl_distance, b2sl_distance
    ], axis=-1)

    # player <-> ball, goal, back line, side line distance
    p2b_distance = get_distance(both_team[:,np.newaxis,:], ball[np.newaxis,:,:])
    p2g_distance = get_distance(both_team[:,np.newaxis,:], goal[np.newaxis,:,:])
    p2gl_distance = get_line_distance(both_team[:,:1], goal_line_x[np.newaxis,:])
    p2sl_distance = get_line_distance(both_team[:,1:], side_line_y[np.newaxis,:])
    p2bo_distance = np.concatenate([
        p2b_distance, p2g_distance, p2gl_distance, p2sl_distance
    ], axis=-1)

    # player <-> player distance
    p2p_distance = get_distance(both_team[:,np.newaxis,:], both_team[np.newaxis,:,:])

    # apply Multiscale to distances
    #def concat_multiscale(x, scale):
    #    return np.concatenate([x[...,np.newaxis], 1 - multi_scale(x, scale)], axis=-1)

    #distance_scales = [.01, .05, .25, 1.25]
    #b2o_distance = 1 - multi_scale(b2o_distance, distance_scales).reshape(-1)
    #p2bo_distance = 1 - multi_scale(p2bo_distance, distance_scales).reshape(len(both_team), -1)
    #p2p_distance = 1 - multi_scale(p2p_distance, distance_scales).reshape(len(both_team), len(both_team), -1)

    # controlled player information
    control_flag_ = np.array(PLAYER_1HOT[obs['active']], dtype=np.float32)
    control_flag = np.concatenate([control_flag_, np.zeros(len(obs['right_team']))])[...,np.newaxis]

    # controlled status information
    DIR = [
        [-1, 0], [-.707, -.707], [0,  1], [ .707, -.707],  # L, TL, T, TR
        [ 1, 0], [ .707,  .707], [0, -1], [-.707,  .707]   # R, BR, B, BL
    ]
    sticky_direction = DIR[obs['sticky_actions'][:8].index(1)] if 1 in obs['sticky_actions'][:8] else [0, 0]
    sticky_flags = obs['sticky_actions'][8:]

    control_features = np.concatenate([
        sticky_direction,
        sticky_flags,
    ]).astype(np.float32)

    # Match state
    if obs['steps_left'] > info['half_step']:
        steps_left_half = obs['steps_left'] - info['half_step']
    else:
        steps_left_half = obs['steps_left']
    match_features = np.concatenate([
        multi_scale(obs['score'], [1, 3]).ravel(),
        multi_scale(obs['score'][0] - obs['score'][1], [1, 3]),
        multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),
        multi_scale(steps_left_half, [10, 100, 1000, 10000]),
        ball_owned_team,
    ]).astype(np.float32)

    mode_index = np.array([obs['game_mode']], dtype=np.int32)

    # Super Mini Map
    # SMM_WIDTH = 96 #// 3
    # SMM_HEIGHT = 72 #// 3
    # SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

    # # Normalized minimap coordinates
    # MINIMAP_NORM_X_MIN = -1.0
    # MINIMAP_NORM_X_MAX = 1.0
    # MINIMAP_NORM_Y_MIN = -1.0 / 2.25
    # MINIMAP_NORM_Y_MAX = 1.0 / 2.25

    # _MARKER_VALUE = 1  # 255

    # def get_smm_layers(config):
    #     return SMM_LAYERS

    # def mark_points(frame, points):
    #     """Draw dots corresponding to 'points'.
    #     Args:
    #       frame: 2-d matrix representing one SMM channel ([y, x])
    #       points: a list of (x, y) coordinates to be marked
    #     """
    #     for p in range(len(points) // 2):
    #         x = int((points[p * 2] - MINIMAP_NORM_X_MIN) /
    #                 (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1])
    #         y = int((points[p * 2 + 1] - MINIMAP_NORM_Y_MIN) /
    #                 (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0])
    #         x = max(0, min(frame.shape[1] - 1, x))
    #         y = max(0, min(frame.shape[0] - 1, y))
    #         frame[y, x] = _MARKER_VALUE

    # def generate_smm(observation, config=None,
    #                  channel_dimensions=(SMM_WIDTH, SMM_HEIGHT)):
    #     """Returns a list of minimap observations given the raw features for each
    #     active player.
    #     Args:
    #       observation: raw features from the environment
    #       config: environment config
    #       channel_dimensions: resolution of SMM to generate
    #     Returns:
    #       (N, H, W, C) - shaped np array representing SMM. N stands for the number of
    #       players we are controlling.
    #     """
    #     frame = np.zeros((len(observation), channel_dimensions[1],
    #                       channel_dimensions[0], len(get_smm_layers(config))),
    #                       dtype=np.uint8)

    #     for o_i, o in enumerate(observation):
    #         for index, layer in enumerate(get_smm_layers(config)):
    #             assert layer in o
    #             if layer == 'active':
    #                 if o[layer] == -1:
    #                     continue
    #                 mark_points(frame[o_i, :, :, index],
    #                             np.array(o['left_team'][o[layer]]).reshape(-1))
    #             else:
    #                 mark_points(frame[o_i, :, :, index], np.array(o[layer]).reshape(-1))
    #     return frame

    # smm = generate_smm([obs]).transpose(3, 1, 2, 0).squeeze(3).astype(np.float32)

    # ACTION_1HOT = np.eye(19)
    # action_history = np.stack([ACTION_1HOT[a] for a in action_history]).astype(np.float32)
    action_history = np.array(action_history, dtype=np.int32)[..., None]

    return {
        # features
        'ball': ball_features,
        'match': match_features,
        'player': {
            'self': left_team_features,
            'opp': right_team_features
        },
        'control': control_features,
        'player_index': {
            'self': left_team_indice,
            'opp': right_team_indice
        },
        'mode_index': mode_index,
        'control_flag': control_flag,
        # distances
        'distance': {
            'p2p': p2p_distance,
            'p2bo': p2bo_distance,
            'b2o': b2o_distance
        },
        # CNN
        'cnn_feature': cnn_feature,
        # SuperMiniMap
        #'smm': smm,
        'action_history': action_history
    }


KICK_ACTIONS = {
    Action.LongPass: 20,
    Action.HighPass: 28,
    Action.ShortPass: 36,
    Action.Shot: 44,
}


class Environment:
    ACTION_LEN = 19 + 4 * 8
    ACTION_IDX = list(range(ACTION_LEN))

    def __init__(self, args={}):
        self.env_map = {}
        self.env = None
        self.limit_steps = args.get('limit_steps', 100000)
        self.frame_skip = args.get('frame_skip', 0)
        self.reset_common()

    def reset_common(self):
        self.finished = False
        self.prev_score = [0, 0]
        self.reset_flag = False
        self.checkpoint = [
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]]
        self.states = []
        self.half_step = 1500
        self.reserved_action = [None, None]

    def reset(self, args={}):
        if len(self.env_map) == 0:
            from gfootball.env import football_action_set
            from gfootball.env.wrappers import Simple115StateWrapper
            from kaggle_environments import make

            self.ACTION_STR = football_action_set.action_set_v1
            self.ACTION2STR = {i: j for i, j in enumerate(football_action_set.action_set_v1)}
            self.STR2ACTION = {j: i for i, j in self.ACTION2STR.items()}

#             self.env_map[3000] = make("football", configuration={"scenario_name": "11_vs_11_kaggle"})
#             self.env_map[1000] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_1000_500"})
#             self.env_map[500] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_500_250"})
#             self.env_map[9999] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_random"})
#             self.env_map[99999] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_random_long"})

            self.env_map["real"] = make("football", configuration={"scenario_name": "11_vs_11_kaggle"})
            self.env_map["eval"] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_1000_500"})
            self.env_map["train"] = make("football", configuration={"scenario_name": "11_vs_11_kaggle_train"})

        # decide limit steps
#         if args.get('role', {}) == 'e':
#             self.env = self.env_map[1000]
#         else:
#             limit_rate = args.get('limit_rate', 1.0)
#             if limit_rate > 0.9:
#                 self.env = self.env_map[3000]
#             elif limit_rate >= 0:
#                 self.env = self.env_map[99999]

        role = args.get('role', '')
        limit_rate = args.get('limit_rate', 1)
        if role == 'g':
            self.env = self.env_map['train' if limit_rate < 0.95 else 'real']
        elif role == 'e':
            self.env = self.env_map['eval']
        else:
            self.env = self.env_map['real']

        state = self.env.reset()
        self.resets_info(state)

    def resets_info(self, state):
        self.reset_common()
        state = copy.deepcopy(state)
        state = [self._preprocess_state(s) for s in state]
        self.states.append(state)
        self.half_step = state[0]['observation']['players_raw'][0]['steps_left'] // 2

    def reset_info(self, state):
        self.resets_info(state)

    def chance(self):
        pass

    def action2str(self, a: int):
        # return self.ACTION2STR[a]
        return str(a)

    def str2action(self, s: str):
        # return self.STR2ACTION[s]
        return int(s)

    def plays(self, actions):
        self._plays(actions)

    def _plays(self, actions):
        # state transition function
        # action is integer (0 ~ 18)
        actions = copy.deepcopy(actions)
        for i, res_action in enumerate(self.reserved_action):
            if res_action is not None:
                actions[i] = res_action

        # augmented action to atomic action
        for i, action in enumerate(actions):
            atomic_a, reserved_a = self.special_to_actions(action)
            actions[i] = atomic_a
            self.reserved_action[i] = reserved_a

        # step environment
        state = self.env.step([[actions[0]], [actions[1]]])
        state = copy.deepcopy(state)
        state = [self._preprocess_state(s) for s in state]
        self.states.append(state)

        # update status
        if state[0]['status'] == 'DONE' or len(self.states) > self.limit_steps:
            self.finished = True

    def plays_info(self, state):
        # state stansition function as an agent
        state = copy.deepcopy(state)
        state = [self._preprocess_state(s) for s in state]
        self.states.append(state)

    def play_info(self, state):
        self.plays_info(state)

    def diff_info(self):
        return self.states[-1]

    def turns(self):
        return self.players()

    def players(self):
        return [0, 1]

    def terminal(self):
        # check whether the state is terminal
        return self.finished

    def reward(self):
        prev_score = self.prev_score
        score = self.score()

        rs = []
        scored_player = None
        for p in self.players():
            r = 1.0 * (score[p] - prev_score[p]) - 1.0 * (score[1 - p] - prev_score[1 - p])
            rs.append(r)
            if r != 0:
                self.reset_flag = True
                scored_player = p

        self.prev_score = self.score()
        return rs

        def get_goal_distance(xy1):
            return (((xy1 - np.array([1, 0])) ** 2).sum(axis=-1)) ** 0.5

        # checkpoint reward (https://arxiv.org/pdf/1907.11180.pdf)
        checkpoint_reward = []
        for p in self.players():
            obs = self.raw_observation(p)['players_raw'][0]
            ball_owned_team = obs['ball_owned_team']
            if ball_owned_team == p and len(self.checkpoint[p]) != 0:
                ball = obs['ball'][:2]
                goal_distance = get_goal_distance(ball)
                if goal_distance < self.checkpoint[p][0]:
                    cr = 0
                    for idx, c in enumerate(self.checkpoint[p]):
                        if goal_distance < c:
                            cr += 0.1
                        else:
                            break
                    self.checkpoint[p] = self.checkpoint[p][idx:]
                    checkpoint_reward.append(cr)
                else:
                    checkpoint_reward.append(0)
            else:
                checkpoint_reward.append(0)

        if scored_player is not None:
            checkpoint_reward[scored_player] += len(self.checkpoint[scored_player]) * 0.1 # add remain reward when scoring (0.05 per checkpoint)
            self.checkpoint[scored_player] = []

        return [rs[p] + checkpoint_reward[p] for p in self.players()]

    def is_reset_state(self):
        if self.reset_flag:
            self.reset_flag = False
            return True
        return False

    def score(self):
        if len(self.states) == 0:
            return [0, 0]
        obs = self.states[-1]
        return [
            obs[0]['observation']['players_raw'][0]['score'][0],
            obs[1]['observation']['players_raw'][0]['score'][0]
        ]

    def outcome(self):
        if len(self.states) == 0:
            return [0, 0]
        scores = self.score()
        if scores[0] > scores[1]:
            score_diff = scores[0] - scores[1]
            outcome_tanh = np.tanh(score_diff ** 0.8)
            return [outcome_tanh, -outcome_tanh]
        elif scores[0] < scores[1]:
            score_diff = scores[1] - scores[0]
            outcome_tanh = np.tanh(score_diff ** 0.8)
            return [-outcome_tanh, outcome_tanh]
        return [0, 0]

    def legal_actions(self, player):
        # legal action list
        all_actions = [i for i in copy.copy(self.ACTION_IDX) if i != 19]

        if len(self.states) == 0:
            return all_actions

        # obs from view of the player
        obs = self.raw_observation(player)['players_raw'][0]
        # Illegal actions
        illegal_actions = set()
        # You have a ball?
        ball_owned_team = obs['ball_owned_team']
        if ball_owned_team != 0:  # not owned or free
            illegal_actions.add(int(Action.LongPass))
            illegal_actions.add(int(Action.HighPass))
            illegal_actions.add(int(Action.ShortPass))
            illegal_actions.add(int(Action.Shot))
            illegal_actions.add(int(Action.Dribble))
            for d in range(8):
                illegal_actions.add(KICK_ACTIONS[Action.LongPass] + d)
                illegal_actions.add(KICK_ACTIONS[Action.HighPass] + d)
                illegal_actions.add(KICK_ACTIONS[Action.ShortPass] + d)
                illegal_actions.add(KICK_ACTIONS[Action.Shot] + d)
        else:  # owned
            illegal_actions.add(int(Action.Slide))

        # Already sticky action?
        sticky_actions = obs['sticky_actions']
        if type(sticky_actions) == set:
            sticky_actions = [0] * 10

        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
            illegal_actions.add(int(Action.ReleaseSprint))

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
            illegal_actions.add(int(Action.ReleaseDribble))

        if 1 not in sticky_actions[:8]:
            illegal_actions.add(int(Action.ReleaseDirection))

        return [a for a in all_actions if a not in illegal_actions]

    def action_length(self):
        # maximum size of policy (it determines output size of policy function)
        return self.ACTION_LEN

    def raw_observation(self, player):
        if len(self.states) > 0:
            return self.states[-1][player]['observation']
        else:
            return OBS_TEMPLATE

    def observation(self, player):
        # input feature for neural nets
        info = {'half_step': self.half_step}
        return feature_from_states(self.states, info, player)

    def _preprocess_state(self, player_state):
        if player_state is None:
            return player_state

        # in ball-dead state, set ball owned player and team
        o = player_state['observation']['players_raw'][0]
        mode = o['game_mode']
        if mode == GameMode.FreeKick or \
            mode == GameMode.Corner or \
            mode == GameMode.Penalty or \
            mode == GameMode.GoalKick:
            # find nearest player and team
            def dist(xy1, xy2):
                return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5
            team_player_position = [(0, i, p) for i, p in enumerate(o['left_team'])] + \
                [(1, i, p) for i, p in enumerate(o['right_team'])]
            distances = [(t[0], t[1], dist(t[2], o['ball'][:2])) for t in team_player_position]
            distances = sorted(distances, key=lambda x: x[2])
            #print(mode, [t[2] for t in distances])
            #print(o['ball_owned_team'], o['ball_owned_player'], '->', distances[0][0], distances[0][1])
            #input()
            o['ball_owned_team'] = distances[0][0]
            o['ball_owned_player'] = distances[0][1]

        # in the beginning, fill actions with 0
        if len(player_state['action']) == 0:
            player_state['action'].append(0)

        return player_state

    def special_to_actions(self, saction):
        if not 0 <= saction < 52:
            return [0, None]
        for a, index in KICK_ACTIONS.items():
            if index <= saction < index + 8:
                return [a, Action(saction - index + 1)]
        return [saction, None]

    '''def action_to_specials(self, action):
        p = np.zeros(self.action_length())
        p[action] = 1

        sticky_direction =


        if action == Action.LongPass:
            return

        return p / p.sum()'''

    def funcname(self, parameter_list):
        """
        docstring
        """
        pass



if __name__ == '__main__':
    e = Environment()
    net = e.net()(e)
    net.eval()
    for _ in range(1):
        e.reset()
        o = e.observation(0)
        net.inference(o, None)
        while not e.terminal():
            # print(e)
            _ = e.observation(0)
            _ = e.observation(1)
            print(e.env.configuration.episodeSteps)
            print(e.raw_observation(0)['players_raw'][0]['steps_left'])
            action_list = [0, 0]
            action_list[0] = random.choice(e.legal_actions(0))
            action_list[1] = e.rule_based_action_C(1)
            print(len(e.states), action_list)
            e.plays(action_list)
            print(e.checkpoint)
            print(e.reward())
        print(e)
        print(e.score())
        print(e.outcome())
