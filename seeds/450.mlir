module {
  func.func @main(%arg0: tensor<16xi16>, %arg1: tensor<1xi16>, %arg2: tensor<12x7x63x29xf32>) -> (tensor<16xi16>, tensor<12x7x63x29xi1>, tensor<12x7x63x29xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<16xi16>, tensor<1xi16>) -> tensor<16xi16>
    %1 = tosa.sigmoid %arg2 : (tensor<12x7x63x29xf32>) -> tensor<12x7x63x29xf32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<16xi16>, tensor<16xi16>) -> tensor<16xi16>
    %3 = tosa.bitwise_xor %2, %0 : (tensor<16xi16>, tensor<16xi16>) -> tensor<16xi16>
    %4 = tosa.greater_equal %1, %1 : (tensor<12x7x63x29xf32>, tensor<12x7x63x29xf32>) -> tensor<12x7x63x29xi1>
    %5 = tosa.sub %1, %1 : (tensor<12x7x63x29xf32>, tensor<12x7x63x29xf32>) -> tensor<12x7x63x29xf32>
    return %3, %4, %5 : tensor<16xi16>, tensor<12x7x63x29xi1>, tensor<12x7x63x29xf32>
  }
}
