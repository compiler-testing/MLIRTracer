module {
  func.func @main(%arg0: tensor<76x53x21x7xi8>, %arg1: tensor<79x83xf32>, %arg2: tensor<71x22x82xi1>, %arg3: tensor<85x41x74x79x91xi32>, %arg4: tensor<85x41x1x1x91xi32>) -> (tensor<76x159x63x21xi8>, tensor<71x22x1xi1>, tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi1>, tensor<85x41x74x79x91xi32>, tensor<79x83xf32>, tensor<79x83xf32>, tensor<79x1xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3, 3, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<76x53x21x7xi8>, !tosa.shape<4>) -> tensor<76x159x63x21xi8>
    %1 = tosa.log %arg1 : (tensor<79x83xf32>) -> tensor<79x83xf32>
    %2 = tosa.exp %1 : (tensor<79x83xf32>) -> tensor<79x83xf32>
    %3 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<71x22x82xi1>) -> tensor<71x22x1xi1>
    %4 = tosa.intdiv %arg3, %arg4 : (tensor<85x41x74x79x91xi32>, tensor<85x41x1x1x91xi32>) -> tensor<85x41x74x79x91xi32>
    %5 = tosa.bitwise_and %4, %4 : (tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi32>) -> tensor<85x41x74x79x91xi32>
    %6 = tosa.sub %5, %4 : (tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi32>) -> tensor<85x41x74x79x91xi32>
    %7 = tosa.pow %2, %2 : (tensor<79x83xf32>, tensor<79x83xf32>) -> tensor<79x83xf32>
    %8 = tosa.clamp %5 {min_val = -51 : i32, max_val = -19 : i32} : (tensor<85x41x74x79x91xi32>) -> tensor<85x41x74x79x91xi32>
    %9 = tosa.bitwise_xor %5, %5 : (tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi32>) -> tensor<85x41x74x79x91xi32>
    %10 = tosa.greater %9, %9 : (tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi32>) -> tensor<85x41x74x79x91xi1>
    %11 = tosa.logical_not %10 : (tensor<85x41x74x79x91xi1>) -> tensor<85x41x74x79x91xi1>
    %12 = tosa.abs %4 : (tensor<85x41x74x79x91xi32>) -> tensor<85x41x74x79x91xi32>
    %13 = tosa.sigmoid %7 : (tensor<79x83xf32>) -> tensor<79x83xf32>
    %14 = tosa.sigmoid %7 : (tensor<79x83xf32>) -> tensor<79x83xf32>
    %15 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<79x83xf32>) -> tensor<79x1xf32>
    return %0, %3, %6, %8, %11, %12, %13, %14, %15 : tensor<76x159x63x21xi8>, tensor<71x22x1xi1>, tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi32>, tensor<85x41x74x79x91xi1>, tensor<85x41x74x79x91xi32>, tensor<79x83xf32>, tensor<79x83xf32>, tensor<79x1xf32>
  }
}
