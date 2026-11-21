import unittest
from benchmark.planning import dist,line_is_free,safe_route,direct_route,split_path,path_is_free

class TestPlanning(unittest.TestCase):
    def test_distance(self): self.assertAlmostEqual(dist((0,0),(3,4)),5)
    def test_direct_blocked(self): self.assertFalse(line_is_free((-.2,0),(.2,0),[(0,0,.05)]))
    def test_safe_route(self):
        r=safe_route((-.25,0),(.25,0),[(0,0,.05)])
        self.assertGreaterEqual(len(r),3); self.assertTrue(path_is_free(r,[(0,0,.05)]))
    def test_split(self):
        a,b=split_path([(0,0),(1,0)],.35)
        self.assertAlmostEqual(a[-1][0],.35,places=5); self.assertAlmostEqual(b[0][0],.35,places=5)

if __name__=='__main__':unittest.main()
