module {
  func.func @main(%arg0: tensor<71x12x65x35xi64>, %arg1: tensor<71x12x1x1xi64>, %arg2: tensor<74x54x56x33x87xi64>, %arg3: tensor<74x54x56x33x1xi64>) -> (tensor<71x12x65x35xi1>, tensor<75168x17094xi1>, tensor<74x54x56x66x87xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<71x12x65x35xi64>, tensor<71x12x1x1xi64>) -> tensor<71x12x65x35xi1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<74x54x56x33x87xi64>, tensor<74x54x56x33x1xi64>) -> tensor<74x54x56x33x87xi64>
    %2 = tosa.logical_and %0, %0 : (tensor<71x12x65x35xi1>, tensor<71x12x65x35xi1>) -> tensor<71x12x65x35xi1>
    %3 = tosa.concat %1, %1 {axis = 3 : i32} : (tensor<74x54x56x33x87xi64>, tensor<74x54x56x33x87xi64>) -> tensor<74x54x56x66x87xi64>
    %4 = tosa.bitwise_or %3, %3 : (tensor<74x54x56x66x87xi64>, tensor<74x54x56x66x87xi64>) -> tensor<74x54x56x66x87xi64>
    %5 = tosa.equal %4, %3 : (tensor<74x54x56x66x87xi64>, tensor<74x54x56x66x87xi64>) -> tensor<74x54x56x66x87xi1>
    %r_6 = tosa.const_shape {values = dense<[ 75168, 17094 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.reshape %5, %r_6 : (tensor<74x54x56x66x87xi1>, !tosa.shape<2>) -> tensor<75168x17094xi1>
    %7 = tosa.bitwise_or %4, %3 : (tensor<74x54x56x66x87xi64>, tensor<74x54x56x66x87xi64>) -> tensor<74x54x56x66x87xi64>
    %8 = tosa.greater_equal %7, %7 : (tensor<74x54x56x66x87xi64>, tensor<74x54x56x66x87xi64>) -> tensor<74x54x56x66x87xi1>
    return %2, %6, %8 : tensor<71x12x65x35xi1>, tensor<75168x17094xi1>, tensor<74x54x56x66x87xi1>
  }
}
