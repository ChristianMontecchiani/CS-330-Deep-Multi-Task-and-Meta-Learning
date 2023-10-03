#!/usr/bin/env python3
import unittest
import random
import sys
import copy
import argparse
import inspect
import collections
import os
import pickle
import gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import torch
import torch.nn as nn
from dataset import get_movielens_dataset
from multitask import MultitaskModel
from evaluation import mrr_score, mse_score
from utils import fix_random_seeds

# Import student protonet
import models

if os.path.exists("./models"):
    model_path = "./models"
else:
    model_path = "."

#########
# TESTS #
#########

class Test_1a(GradedTestCase):
    def setUp(self):
        self.embedding_dim = 32
        self.num_users = 944
        self.num_items = 1683
        self.layer_sizes = [96, 64]
        self.multitask_net = models.MultiTaskNet
        self.sol_multitask_net = self.run_with_solution_if_possible(models, lambda sub_or_sol:sub_or_sol.MultiTaskNet)
    
    @graded()
    def test_0(self):
        """1a-0-basic:  single user and item embedding shape"""
        U, Q = self.multitask_net.init_shared_user_and_item_embeddings(self, self.num_users, self.num_items, self.embedding_dim)
        if U == None or Q == None:
            self.assertTrue(False, 'init_shared_user_and_item_embeddings not implemented')
        self.assertTrue(U.weight.shape == torch.Size([self.num_users, self.embedding_dim]), 'incorrect tensor shape for self.U')
        self.assertTrue(Q.weight.shape == torch.Size([self.num_items, self.embedding_dim]), 'incorrect tensor shape for self.Q')

    @graded()
    def test_1(self):
        """1a-1-basic: multiple user and item embedding shape"""
        U_reg, Q_reg, U_fact, Q_fact = self.multitask_net.init_separate_user_and_item_embeddings(self, self.num_users, self.num_items, self.embedding_dim)
        if U_reg == None or Q_reg == None or U_fact == None or Q_fact == None:
            self.assertTrue(False, 'init_separate_user_and_item_embeddings not implemented')
        self.assertTrue(U_reg.weight.shape == torch.Size([self.num_users, self.embedding_dim]), 'incorrect tensor shape for self.U_reg')
        self.assertTrue(Q_reg.weight.shape == torch.Size([self.num_items, self.embedding_dim]), 'incorrect tensor shape for self.Q_reg')
        self.assertTrue(U_fact.weight.shape == torch.Size([self.num_users, self.embedding_dim]), 'incorrect tensor shape for self.U_fact')
        self.assertTrue(Q_fact.weight.shape == torch.Size([self.num_items, self.embedding_dim]), 'incorrect tensor shape for self.Q_fact')
    
    @graded()
    def test_2(self):
        """1a-2-basic: user and item bias terms shape"""
        B = self.multitask_net.init_item_bias(self, self.num_users, self.num_items)
        if B == None:
            self.assertTrue(False, 'init_item_bias not implemented')
        self.assertTrue(B.weight.shape == torch.Size([self.num_items, 1]), 'incorrect tensor shape for self.B')

class Test_1b(GradedTestCase):
    def setUp(self):
        self.embedding_dim = 32
        self.num_users = 944
        self.num_items = 1683
        self.layer_sizes = [96, 64]
        self.multitask_net = models.MultiTaskNet
        self.sol_multitask_net = self.run_with_solution_if_possible(models, lambda sub_or_sol:sub_or_sol.MultiTaskNet)
    
    @graded()
    def test_0(self):
        """1b-0-basic: verifying mlp layers size and type at each layer"""
        mlp_layers = self.multitask_net.init_mlp_layers(self, self.layer_sizes)
        if mlp_layers == None:
            self.assertTrue(False, 'init_mlp_layers not implemented')
        self.assertTrue(len(mlp_layers) == 3, 'incorrect self.mlp_layers size')
        self.assertTrue(type(mlp_layers[0]).__name__ == 'Linear', 'incorrect type for first layer of self.mlp_layers')
        self.assertTrue(type(mlp_layers[1]).__name__ == 'ReLU', 'incorrect type for second layer of self.mlp_layers')
        self.assertTrue(type(mlp_layers[2]).__name__ == 'Linear', 'incorrect type for third layer of self.mlp_layers')
    
    
class Test_2(GradedTestCase):
    def setUp(self):

        fix_random_seeds()

        self.multitask_net_with_sharing = models.MultiTaskNet(944, 1683, 32, [96, 64], True)
        self.multitask_net_without_sharing = models.MultiTaskNet(944, 1683, 32, [96, 64], False)

        self.dataset = get_movielens_dataset(variant='100K')
        self.train, self.test = self.dataset.random_train_test_split(test_fraction=0.05)

        self.model_with_sharing = MultitaskModel(interactions=self.train,
                                                representation=self.multitask_net_with_sharing,
                                                factorization_weight=0.5,
                                                regression_weight=0.5)
        self.model_without_sharing = MultitaskModel(interactions=self.train,
                                                representation=self.multitask_net_without_sharing,
                                                factorization_weight=0.5,
                                                regression_weight=0.5)

        self.sol_multitask_net = self.run_with_solution_if_possible(models, lambda sub_or_sol:sub_or_sol.MultiTaskNet)
        self.num_users = torch.randint(1, 500, [256])
        self.num_items = torch.randint(1, 500, [256])
        self.total_epochs = 5
    
    @graded()
    def test_0(self):
        """2-0-basic: check prediction and score shapes for forward_with_embedding_sharing"""
        predictions, score = self.multitask_net_with_sharing.forward_with_embedding_sharing(self.num_users, self.num_items)
        if predictions == None or score == None:
            self.assertTrue(False, 'forward_with_embedding_sharing not implemented')
        self.assertTrue(predictions.shape == torch.Size([256,]), 'incorrect tensor shape for predictions')
        self.assertTrue(score.shape == torch.Size([256,]), 'incorrect tensor shape for score')
    
    @graded()
    def test_1(self):
        """2-1-basic: check prediction and score shapes for forward_without_embedding_sharing"""
        predictions, score = self.multitask_net_without_sharing.forward_without_embedding_sharing(self.num_users, self.num_items)
        if predictions == None or score == None:
            self.assertTrue(False, 'forward_without_embedding_sharing not implemented')
        self.assertTrue(predictions.shape == torch.Size([256,]), 'incorrect tensor shape for predictions')
        self.assertTrue(score.shape == torch.Size([256,]), 'incorrect tensor shape for score')
    
    @graded(timeout=30)
    def test_2(self):
        """2-2-basic: mrr, mse accuracy for forward_with_embedding_sharing"""

        fix_random_seeds()

        mrr = mse = None
        for epoch in range(self.total_epochs):
            factorization_loss, score_loss, joint_loss = self.model_with_sharing.fit(self.train)
            mrr = mrr_score(self.model_with_sharing, self.test, self.train)
            mse = mse_score(self.model_with_sharing, self.test)
        # self.assertAlmostEqual(mrr, 0.01, delta=0.01, msg="mrr not converging to expected values")
        # self.assertAlmostEqual(mse, 13.06, delta=0.5, msg="mse not converging to expected values")
        self.assertAlmostEqual(mrr, 0.06, delta=0.02, msg="mrr not converging to expected values")
        self.assertAlmostEqual(mse, 0.9, delta=0.05, msg="mse not converging to expected values")
    
    @graded(timeout=30)
    def test_3(self):
        """2-3-basic: mrr, mse accuracy for forward_without_embedding_sharing"""

        fix_random_seeds()

        mrr = mse = None
        for epoch in range(self.total_epochs):
            factorization_loss, score_loss, joint_loss = self.model_without_sharing.fit(self.train)
            mrr = mrr_score(self.model_without_sharing, self.test, self.train)
            mse = mse_score(self.model_without_sharing, self.test)
        # print(mrr)
        # print(mse)
        # self.assertAlmostEqual(mrr, 0.01, delta=0.01, msg="mrr not converging to expected values")
        # self.assertAlmostEqual(mse, 12.96, delta=0.5, msg="mse not converging to expected values")
        self.assertAlmostEqual(mrr, 0.08, delta=0.02, msg="mrr not converging to expected values")
        self.assertAlmostEqual(mse, 0.9, delta=0.05, msg="mse not converging to expected values")
    
def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)