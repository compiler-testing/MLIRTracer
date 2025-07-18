module {
  func.func @main(%arg0: tensor<28x63x29xi8>, %arg1: tensor<95xf32>) -> (tensor<28x63x1xi8>, tensor<95xf32>) {
    %0 = tosa.clamp %arg0 {min_val = -21 : i8, max_val = 8 : i8} : (tensor<28x63x29xi8>) -> tensor<28x63x29xi8>
    %1 = tosa.clz %0 : (tensor<28x63x29xi8>) -> tensor<28x63x29xi8>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<28x63x29xi8>) -> tensor<28x63x1xi8>
    %3 = tosa.abs %2 : (tensor<28x63x1xi8>) -> tensor<28x63x1xi8>
    %4 = tosa.tanh %arg1 : (tensor<95xf32>) -> tensor<95xf32>
    return %3, %4 : tensor<28x63x1xi8>, tensor<95xf32>
  }
}
