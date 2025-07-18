module {
  func.func @main(%arg0: tensor<56x6x96xi8>, %arg1: tensor<1x6x1xi8>, %arg2: tensor<37xi32>, %arg3: tensor<1xi32>, %arg4: tensor<73x80x51x50x71xi32>, %arg5: tensor<73x1x1x50x1xi32>, %arg6: tensor<98xf32>, %arg7: tensor<98xf32>) -> (tensor<1xi1>, tensor<56x6x1xi1>, tensor<73x80x51x50x71xi1>, tensor<98xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<56x6x96xi8>, tensor<1x6x1xi8>) -> tensor<56x6x96xi1>
    %1 = tosa.clz %0 : (tensor<56x6x96xi1>) -> tensor<56x6x96xi1>
    %2 = tosa.bitwise_xor %1, %0 : (tensor<56x6x96xi1>, tensor<56x6x96xi1>) -> tensor<56x6x96xi1>
    %3 = tosa.equal %arg2, %arg3 : (tensor<37xi32>, tensor<1xi32>) -> tensor<37xi1>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<37xi1>) -> tensor<1xi1>
    %5 = tosa.reduce_min %2 {axis = 2 : i32} : (tensor<56x6x96xi1>) -> tensor<56x6x1xi1>
    %6 = tosa.equal %arg4, %arg5 : (tensor<73x80x51x50x71xi32>, tensor<73x1x1x50x1xi32>) -> tensor<73x80x51x50x71xi1>
    %7 = tosa.equal %arg6, %arg7 : (tensor<98xf32>, tensor<98xf32>) -> tensor<98xi1>
    return %4, %5, %6, %7 : tensor<1xi1>, tensor<56x6x1xi1>, tensor<73x80x51x50x71xi1>, tensor<98xi1>
  }
}
