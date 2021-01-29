"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
import numpy as np
import oneflow as flow
import oneflow.typing as tp

from scipy.sparse import csr_matrix


def GenerateTest(
    test_case, a_csrRowOffsets, a_csrColInd, a_csrValues, a_rows, a_cols, b
):
    @flow.global_function()
    def SpmmCSRJob(
        a_csrRowOffsets: tp.Numpy.Placeholder((5,), dtype=flow.int32),
        a_csrColInd: tp.Numpy.Placeholder((9,), dtype=flow.int32),
        a_csrValues: tp.Numpy.Placeholder((9,), dtype=flow.float32),
        b: tp.Numpy.Placeholder((4, 3), dtype=flow.float32),
    ) -> tp.Numpy:
        with flow.scope.placement("gpu", "0:0"):
            return flow.spmm_csr(
                a_csrRowOffsets, a_csrColInd, a_csrValues, a_rows, a_cols, b
            )

    y = SpmmCSRJob(a_csrRowOffsets, a_csrColInd, a_csrValues, b)
    x = (
        csr_matrix(
            (a_csrValues, a_csrColInd, a_csrRowOffsets), shape=(a_rows, a_cols)
        )
        * b
    )
    test_case.assertTrue(np.array_equal(y, x))


@flow.unittest.skip_unless_1n1d()
class TestSpmmCSR(flow.unittest.TestCase):
    def test_naive(test_case):
        a_csrRowOffsets = np.array([0, 3, 4, 7, 9], dtype=np.int32)
        a_csrColInd = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
        a_csrValues = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32
        )
        a_rows = 4
        a_cols = 4
        b = np.array(
            [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]],
            dtype=np.float32,
        )
        GenerateTest(
            test_case, a_csrRowOffsets, a_csrColInd, a_csrValues, a_rows, a_cols, b
        )


if __name__ == "__main__":
    unittest.main()
