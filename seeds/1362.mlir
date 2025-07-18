module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<70x29x94x12xf32>) -> (tensor<i1>, tensor<70x29x94x12xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %1 = tosa.tanh %arg2 : (tensor<70x29x94x12xf32>) -> tensor<70x29x94x12xf32>
    %2 = tosa.minimum %1, %1 : (tensor<70x29x94x12xf32>, tensor<70x29x94x12xf32>) -> tensor<70x29x94x12xf32>
    %3 = tosa.bitwise_xor %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.identity %1 : (tensor<70x29x94x12xf32>) -> tensor<70x29x94x12xf32>
    %5 = tosa.greater %2, %4 : (tensor<70x29x94x12xf32>, tensor<70x29x94x12xf32>) -> tensor<70x29x94x12xi1>
    return %3, %5 : tensor<i1>, tensor<70x29x94x12xi1>
  }
}
