module {
  func.func @main(%arg0: tensor<27x1x94x31x54x22xf32>) -> tensor<10x6x6x1x10x6xf32> {
    %0 = tosa.log %arg0 : (tensor<27x1x94x31x54x22xf32>) -> tensor<27x1x94x31x54x22xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 17, 0, 11, 2, 14, 16 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_1_size = tosa.const_shape {values = dense<[ 10, 6, 1, 6, 10, 6 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<27x1x94x31x54x22xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<10x6x1x6x10x6xf32>
    %2 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0, 1, 3, 2, 4, 5>} : (tensor<10x6x1x6x10x6xf32>) -> tensor<10x6x6x1x10x6xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<10x6x6x1x10x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10x6x6x1x10x6xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<10x6x6x1x10x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10x6x6x1x10x6xf32>
    return %5 : tensor<10x6x6x1x10x6xf32>
  }
}
