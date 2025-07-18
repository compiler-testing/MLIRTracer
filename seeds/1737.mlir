module {
  func.func @main(%arg0: tensor<20x70x95x53xi64>, %arg1: tensor<21x85x28x11xf32>) -> (tensor<7049000xi64>, tensor<21x85x28x11xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<20x70x95x53xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<20x70x95x53xi64>
    %1 = tosa.maximum %0, %0 : (tensor<20x70x95x53xi64>, tensor<20x70x95x53xi64>) -> tensor<20x70x95x53xi64>
    %r_2 = tosa.const_shape {values = dense<[ 7049000 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.reshape %1, %r_2 : (tensor<20x70x95x53xi64>, !tosa.shape<1>) -> tensor<7049000xi64>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<7049000xi64>) -> tensor<7049000xi64>
    %4 = tosa.floor %arg1 : (tensor<21x85x28x11xf32>) -> tensor<21x85x28x11xf32>
    return %3, %4 : tensor<7049000xi64>, tensor<21x85x28x11xf32>
  }
}
