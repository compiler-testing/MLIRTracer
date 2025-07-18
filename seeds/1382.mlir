module {
  func.func @main(%arg0: tensor<36xf32>) -> tensor<10xf32> {
    %0 = tosa.log %arg0 : (tensor<36xf32>) -> tensor<36xf32>
    %1 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0>} : (tensor<36xf32>) -> tensor<36xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<36xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<36xf32>
    %4 = tosa.log %3 : (tensor<36xf32>) -> tensor<36xf32>
    %t_5 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.tile %4, %t_5 : (tensor<36xf32>, !tosa.shape<1>) -> tensor<36xf32>
    %s_6_start = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_6_size = tosa.const_shape {values = dense<[ 5 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<36xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<5xf32>
    %t_7 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.tile %6, %t_7 : (tensor<5xf32>, !tosa.shape<1>) -> tensor<10xf32>
    return %7 : tensor<10xf32>
  }
}
