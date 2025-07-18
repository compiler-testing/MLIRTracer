module {
  func.func @main(%arg0: tensor<50xf32>, %arg1: tensor<66x61x66x22x85x6xi32>, %arg2: tensor<66x1x1x1x1x1xi32>, %arg3: tensor<32x34x19x38x23xi1>, %arg4: tensor<86xi1>) -> (tensor<66x61x66x22x85x6xi32>, tensor<32x34x19x38x23xi1>, tensor<1xf32>, tensor<1xi1>, tensor<1xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<50xf32>) -> tensor<50xf32>
    %1 = tosa.exp %0 : (tensor<50xf32>) -> tensor<50xf32>
    %2 = tosa.logical_left_shift %arg1, %arg2 : (tensor<66x61x66x22x85x6xi32>, tensor<66x1x1x1x1x1xi32>) -> tensor<66x61x66x22x85x6xi32>
    %3 = tosa.reciprocal %1 : (tensor<50xf32>) -> tensor<50xf32>
    %4 = tosa.tanh %3 : (tensor<50xf32>) -> tensor<50xf32>
    %5 = tosa.logical_not %arg3 : (tensor<32x34x19x38x23xi1>) -> tensor<32x34x19x38x23xi1>
    %6 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<86xi1>) -> tensor<1xi1>
    %7 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<50xf32>) -> tensor<1xf32>
    %8 = tosa.clz %6 : (tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.logical_or %8, %8 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.abs %9 : (tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.logical_or %8, %8 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.reduce_product %10 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %2, %5, %7, %11, %12 : tensor<66x61x66x22x85x6xi32>, tensor<32x34x19x38x23xi1>, tensor<1xf32>, tensor<1xi1>, tensor<1xi1>
  }
}
