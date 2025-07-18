module {
  func.func @main(%arg0: tensor<67x17x58xf32>, %arg1: tensor<67x58x45xf32>, %arg2: tensor<69xi1>) -> (tensor<1xi1>, tensor<51255xi1>, tensor<51255xf32>, tensor<1xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<67x17x58xf32>, tensor<67x58x45xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<67x17x45xf32>
    %r_1 = tosa.const_shape {values = dense<[ 51255 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.reshape %0, %r_1 : (tensor<67x17x45xf32>, !tosa.shape<1>) -> tensor<51255xf32>
    %2 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<69xi1>) -> tensor<1xi1>
    %3 = tosa.sub %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.maximum %1, %1 : (tensor<51255xf32>, tensor<51255xf32>) -> tensor<51255xf32>
    %5 = tosa.logical_not %3 : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.greater_equal %4, %1 : (tensor<51255xf32>, tensor<51255xf32>) -> tensor<51255xi1>
    %7 = tosa.minimum %1, %1 : (tensor<51255xf32>, tensor<51255xf32>) -> tensor<51255xf32>
    %s_8_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_8_size = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %8 = tosa.slice %3, %s_8_start, %s_8_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xi1>
    return %5, %6, %7, %8 : tensor<1xi1>, tensor<51255xi1>, tensor<51255xf32>, tensor<1xi1>
  }
}
