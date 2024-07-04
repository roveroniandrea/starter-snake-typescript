import { Tensor } from '@tensorflow/tfjs-node'

export type Experience = {
    stateTensor: Tensor;
    targetQValues: Tensor;
}