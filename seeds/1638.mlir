module {
  func.func @main(%arg0: tensor<57x53x33x60x45x70xi1>, %arg1: tensor<57x53x1x60x1x1xi1>, %arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<63x57x85x57x49xf32>, %arg5: tensor<1x57x1x1x1xf32>, %arg6: tensor<63x94x65xf32>, %arg7: tensor<i32>, %arg8: tensor<i32>) -> (tensor<57x53x33x60x45x70xi1>, tensor<i1>, tensor<63x57x85x57x49xi1>, tensor<63x94x1xf32>, tensor<i32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<57x53x33x60x45x70xi1>, tensor<57x53x1x60x1x1xi1>) -> tensor<57x53x33x60x45x70xi1>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<57x53x33x60x45x70xi1>, tensor<57x53x33x60x45x70xi1>) -> tensor<57x53x33x60x45x70xi1>
    %2 = tosa.greater_equal %arg2, %arg3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %3 = tosa.greater_equal %arg4, %arg5 : (tensor<63x57x85x57x49xf32>, tensor<1x57x1x1x1xf32>) -> tensor<63x57x85x57x49xi1>
    %4 = tosa.reduce_product %arg6 {axis = 2 : i32} : (tensor<63x94x65xf32>) -> tensor<63x94x1xf32>
    %5 = tosa.reverse %4 {axis = 2 : i32} : (tensor<63x94x1xf32>) -> tensor<63x94x1xf32>
    %6 = tosa.intdiv %arg7, %arg8 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %1, %2, %3, %5, %6 : tensor<57x53x33x60x45x70xi1>, tensor<i1>, tensor<63x57x85x57x49xi1>, tensor<63x94x1xf32>, tensor<i32>
  }
}
