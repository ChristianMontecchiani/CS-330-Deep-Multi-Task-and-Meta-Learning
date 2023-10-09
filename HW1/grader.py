#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip, shutil
from graderUtil import graded, CourseTestRunner, GradedTestCase
import torch
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

from main import meta_train_step

# Import submission
import submission

device = torch.device("cpu")

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

def fix_random_seeds(
        seed=123,
        set_system=True,
        set_torch=False):
    """
    Fix random seeds for reproducibility.
    Parameters
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    """
    # set system seed
    if set_system:
        random.seed(seed)
        np.random.seed(seed)

    # set torch seed
    if set_torch:
        torch.manual_seed(seed)

def check_omniglot():
    """
    Check if Omniglot dataset is available.
    """
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )
    assert os.path.isdir("./omniglot_resized"), "Omniglot dataset is not available! Run `python main.py` first to download the dataset!"

#########
# TESTS #
#########


class Test_1(GradedTestCase):

    def setUp(self):
        check_omniglot()
        
        self.K, self.N, self.M, self.B = 1, 2, 784, 128

        self.data_generator = submission.DataGenerator
        self.sol_data_generator = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.DataGenerator)

    @graded(timeout=1)
    def test_0(self):
        """1-0-basic:  Basic test case for testing the output shape."""
        
        # Create student data generator and loader
        data_gen = self.data_generator(
            self.N,
            self.K + 1,
            batch_type="test",
            cache=False
        )

        # Sample once from the student data generator
        images, labels = data_gen._sample()

        # Check that the dimension match for the sampled images
        self.assertTrue(images.shape[0] == self.K+1 and images.shape[1] == self.N and images.shape[2] == self.M, "Issue in DataGenerator._sample function! Please follow all requirements outlined in the function comments and the writeup.")

        # Check that the dimension match for the sampled image labels
        self.assertTrue(labels.shape[0] == self.K+1 and labels.shape[1] == self.N and labels.shape[2] == self.N, "Issue in DataGenerator._sample function! Please follow all requirements outlined in the function comments and the writeup.")

    @graded(timeout=2)
    def test_1(self):
        """1-1-basic: Basic test case for checking the ordering from the support and query sets."""
        
        # Create student data generator and loader
        data_gen = self.data_generator(
            self.N,
            self.K + 1,
            batch_type="test",
            cache=False
        )

        # Create a PyTorch Dataloader based on the student data generator
        test_loader = iter(
            torch.utils.data.DataLoader(
                data_gen,
                batch_size=self.B,
                num_workers=0,
                pin_memory=True,
            )
        )

        # Sample a batch of inputs
        images, labels = next(test_loader)
        images, labels = images.to(device), labels.to(device)

        # Create a fixed order target labels for the sequence set
        target_labels = np.array([])
        for i in range(self.N):
            target_labels = np.append(target_labels, np.eye(N=1, M=self.N, k=i).reshape(-1))
        target_labels = np.tile(target_labels, self.B).reshape(self.B, self.N, self.N)

        # Check that the order of the sequence set is fixed
        self.assertTrue(np.array_equal(labels[:, 0].numpy(), target_labels), "Issue in DataGenerator._sample function! Please follow all requirements outlined in the function comments and the writeup.")

        # Check that the order of the query set is shuffled, i.e. not fixed ordered
        self.assertFalse(np.array_equal(labels[:, self.K].numpy(), target_labels), "Issue in DataGenerator._sample function! Please follow all requirements outlined in the function comments and the writeup.")

    
    
class Test_2(GradedTestCase):

    def setUp(self):

        check_omniglot()

        self.K, self.N, self.B, self.H = 2, 2, 128, 128

        self.data_generator = submission.DataGenerator
        self.sol_data_generator = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.DataGenerator)

        self.mann = submission.MANN
        self.sol_mann = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.MANN)

    @graded(timeout=2)
    def test_0(self):
        """2-0-basic:  Basic test case for testing the MANN model output shape."""

        fix_random_seeds()

        # Create student data generator and loader
        data_gen = self.data_generator(
            self.N,
            self.K + 1,
            batch_type="test",
            cache=False
        )

        # Create a PyTorch Dataloader based on the student data generator
        test_loader = iter(
            torch.utils.data.DataLoader(
                data_gen,
                batch_size=self.B,
                num_workers=0,
                pin_memory=True,
            )
        )

        # Create student model
        model = self.mann(self.N, self.K + 1, self.H)
        model.to(device)

        # Sample a batch of inputs
        images, labels = next(test_loader)
        images, labels = images.to(device), labels.to(device)

        # Do a forward pass and compute the loss
        pred, tls = meta_train_step(images, labels, model, None, eval=True)

        # Check that the dimension match for the predictions
        self.assertTrue(pred.shape[0] == self.B and pred.shape[1] == self.K+1 and pred.shape[2] == self.N and pred.shape[3] == self.N, "Issue in MANN.forward function! Please follow all requirements outlined in the function comments and the writeup.")

    

class Test_3(GradedTestCase):

    def setUp(self):

        self.B = 128


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
