module {
  func.func @main(%arg0: tensor<56x72x91xi64>, %arg1: tensor<19x55x72x51x4x76xf32>, %arg2: tensor<12x96xi1>, %arg3: tensor<12x1xi1>) -> (tensor<56x72x91xi64>, tensor<12x96xi1>, tensor<66x465120x38xf32>) {
    %0 = tosa.clamp %arg0 {min_val = -60 : i64, max_val = -15 : i64} : (tensor<56x72x91xi64>) -> tensor<56x72x91xi64>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<56x72x91xi64>, tensor<56x72x91xi64>) -> tensor<56x72x91xi64>
    %2 = tosa.logical_right_shift %1, %0 : (tensor<56x72x91xi64>, tensor<56x72x91xi64>) -> tensor<56x72x91xi64>
    %3 = tosa.log %arg1 : (tensor<19x55x72x51x4x76xf32>) -> tensor<19x55x72x51x4x76xf32>
    %r_4 = tosa.const_shape {values = dense<[ 66, 465120, 38 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %3, %r_4 : (tensor<19x55x72x51x4x76xf32>, !tosa.shape<3>) -> tensor<66x465120x38xf32>
    %5 = tosa.logical_and %arg2, %arg3 : (tensor<12x96xi1>, tensor<12x1xi1>) -> tensor<12x96xi1>
    %6 = tosa.tanh %4 : (tensor<66x465120x38xf32>) -> tensor<66x465120x38xf32>
    return %2, %5, %6 : tensor<56x72x91xi64>, tensor<12x96xi1>, tensor<66x465120x38xf32>
  }
}
