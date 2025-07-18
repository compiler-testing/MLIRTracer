module {
  func.func @main(%arg0: tensor<30x39x21x8x70x17xi64>, %arg1: tensor<30x39x21x8x70x1xi64>, %arg2: tensor<57xi16>, %arg3: tensor<79x84xf32>) -> (tensor<30x39x21x8x70x17xi1>, tensor<57xi16>, tensor<79x84xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<30x39x21x8x70x17xi64>, tensor<30x39x21x8x70x1xi64>) -> tensor<30x39x21x8x70x17xi1>
    %1 = tosa.abs %0 : (tensor<30x39x21x8x70x17xi1>) -> tensor<30x39x21x8x70x17xi1>
    %2 = tosa.reverse %arg2 {axis = 0 : i32} : (tensor<57xi16>) -> tensor<57xi16>
    %3 = tosa.identity %2 : (tensor<57xi16>) -> tensor<57xi16>
    %4 = tosa.tanh %arg3 : (tensor<79x84xf32>) -> tensor<79x84xf32>
    %5 = tosa.abs %3 : (tensor<57xi16>) -> tensor<57xi16>
    %6 = tosa.bitwise_xor %3, %3 : (tensor<57xi16>, tensor<57xi16>) -> tensor<57xi16>
    %7 = tosa.add %6, %5 : (tensor<57xi16>, tensor<57xi16>) -> tensor<57xi16>
    %8 = tosa.minimum %4, %4 : (tensor<79x84xf32>, tensor<79x84xf32>) -> tensor<79x84xf32>
    %9 = tosa.maximum %8, %8 : (tensor<79x84xf32>, tensor<79x84xf32>) -> tensor<79x84xf32>
    return %1, %7, %9 : tensor<30x39x21x8x70x17xi1>, tensor<57xi16>, tensor<79x84xf32>
  }
}
