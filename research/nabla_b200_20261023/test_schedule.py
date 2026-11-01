"""CPU checks for schedule boundaries, partial batches and independent branches."""
import math
import unittest
from schedule import endpoint_plan, samples_at_step, factor, WARMUP_EPOCHS
from run_all import stages

class ScheduleTests(unittest.TestCase):
    rows = 8786110
    batch = 256
    def test_once_only_warmup(self):
        self.assertEqual(WARMUP_EPOCHS, .3)
        for epochs, expected in [(0,0),(.1,1/3),(.15,.5),(.3,1),(1,1),(3,1),(6,1),(21,1)]:
            self.assertAlmostEqual(factor(epochs*self.rows,self.rows,self.batch),expected)
    def test_exact_full_epochs_and_zero_terminal_lr(self):
        plan = endpoint_plan(self.rows,self.batch)
        self.assertEqual([p['end_step'] for p in plan.values()],[102963,205926,411852,755062])
        for endpoint, spec in plan.items():
            epoch = int(endpoint.split('_')[1])
            self.assertEqual(spec['end_examples'],epoch*self.rows)
            self.assertEqual(spec['effective_epochs'],epoch)
            self.assertEqual(spec['start_step'],math.ceil(.9*spec['end_step']))
            start = samples_at_step(spec['start_step'],self.rows,self.batch)
            self.assertEqual(factor(start,self.rows,self.batch,endpoint),1.)
            self.assertAlmostEqual(factor((start+spec['end_examples'])/2,self.rows,self.batch,endpoint),.5)
            self.assertEqual(factor(spec['end_examples'],self.rows,self.batch,endpoint),0.)
    def test_partial_final_batch(self):
        last = math.ceil(self.rows/self.batch)
        self.assertEqual(samples_at_step(last,self.rows,self.batch),self.rows)
        self.assertEqual(samples_at_step(last+1,self.rows,self.batch),self.rows+256)
        self.assertEqual(samples_at_step(last,self.rows,self.batch)-samples_at_step(last-1,self.rows,self.batch),self.rows%256)
    def test_independent_branch_order_and_resume(self):
        plan = endpoint_plan(self.rows,self.batch)
        work = list(stages(plan))
        self.assertEqual(len(work),8)
        main_last = 0
        executed = 0
        for (main,anchor),(branch,end) in zip(work[::2],work[1::2]):
            self.assertIsNone(main)
            self.assertGreater(anchor,main_last)
            executed += anchor-main_last + end-anchor
            main_last = anchor
            # A decay must not change the LR of the subsequent main run.
            self.assertEqual(factor(samples_at_step(end,self.rows,self.batch),self.rows,self.batch),1.)
        self.assertEqual(executed,827135)

if __name__ == '__main__': unittest.main()
