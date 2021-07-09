import unittest
import recommendation

class ValidatorsTestCase(unittest.TestCase):
    def setUp(self):
        print('..... Preparing Test Data .....')
        self.valid_product = [
           [
                {
                    "productid": "HBV00000NE0T8",
                    "name": "carrefour ayçiçek yağı 5 lt",
                    "score": 1.0
                }
            ]
        ]

    def tearDown(self):
        print('..... Deleting Test Data .....')
        self.valid_product = []

    def test_valid_products(self):
        for address in self.valid_product:
            res = recommendation.results("HBV00000NE0T4")[:1]
            self.assertEqual(res, address)

if  __name__ == '__main__':
    unittest.main()