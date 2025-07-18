module {
  func.func @main(%arg0: tensor<95x36x2xi32>, %arg1: tensor<95x2x48xi32>, %arg2: tensor<50x9x63x16x11xf32>) -> (tensor<50x9x63x16x11xf32>, tensor<50x9x63x16x11xf32>, tensor<95x36x48xi32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<95x36x2xi32>, tensor<95x2x48xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<95x36x48xi32>
    %1 = tosa.rsqrt %arg2 : (tensor<50x9x63x16x11xf32>) -> tensor<50x9x63x16x11xf32>
    %2 = tosa.floor %1 : (tensor<50x9x63x16x11xf32>) -> tensor<50x9x63x16x11xf32>
    %3 = tosa.add %1, %1 : (tensor<50x9x63x16x11xf32>, tensor<50x9x63x16x11xf32>) -> tensor<50x9x63x16x11xf32>
    %4 = tosa.add %3, %3 : (tensor<50x9x63x16x11xf32>, tensor<50x9x63x16x11xf32>) -> tensor<50x9x63x16x11xf32>
    %5 = tosa.pow %4, %3 : (tensor<50x9x63x16x11xf32>, tensor<50x9x63x16x11xf32>) -> tensor<50x9x63x16x11xf32>
    %6 = tosa.bitwise_xor %0, %0 : (tensor<95x36x48xi32>, tensor<95x36x48xi32>) -> tensor<95x36x48xi32>
    return %2, %5, %6 : tensor<50x9x63x16x11xf32>, tensor<50x9x63x16x11xf32>, tensor<95x36x48xi32>
  }
}
