module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<51x57x21x10x63x42xi1>, %arg2: tensor<52xf32>) -> (tensor<i1>, tensor<52xf32>, tensor<21x42x57x10x51x63xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.logical_or %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %2 = tosa.bitwise_not %1 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.arithmetic_right_shift %3, %3 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = "tosa.const"() {values = dense<[2, 5, 1, 3, 0, 4]> : tensor<6xi32>} : () -> tensor<6xi32>
    %6 = tosa.transpose %arg1 {perms = array<i32: 2, 5, 1, 3, 0, 4>} : (tensor<51x57x21x10x63x42xi1>) -> tensor<21x42x57x10x51x63xi1>
    %7 = tosa.sigmoid %arg2 : (tensor<52xf32>) -> tensor<52xf32>
    %8 = tosa.add %6, %6 : (tensor<21x42x57x10x51x63xi1>, tensor<21x42x57x10x51x63xi1>) -> tensor<21x42x57x10x51x63xi1>
    return %4, %7, %8 : tensor<i1>, tensor<52xf32>, tensor<21x42x57x10x51x63xi1>
  }
}
