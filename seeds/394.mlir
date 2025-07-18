module {
  func.func @main(%arg0: tensor<20x13x36x36x37x88xi32>, %arg1: tensor<20x13x36x1x1x88xi32>, %arg2: tensor<89x40x96xi1>, %arg3: tensor<39x18x29x56xf32>, %arg4: tensor<90x70x63x86xf32>, %arg5: tensor<90xf32>) -> (tensor<20x13x36x36x37x88xi32>, tensor<39x89x94x90xf32>, tensor<178x192x1xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<20x13x36x36x37x88xi32>, tensor<20x13x36x1x1x88xi32>) -> tensor<20x13x36x36x37x88xi32>
    %1 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<89x40x96xi1>) -> tensor<89x1x96xi1>
    %t_2 = tosa.const_shape {values = dense<[ 2, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %1, %t_2 : (tensor<89x1x96xi1>, !tosa.shape<3>) -> tensor<178x1x192xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 39, 89, 94, 90>} : (tensor<39x18x29x56xf32>, tensor<90x70x63x86xf32>, tensor<90xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<39x89x94x90xf32>
    %4 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = tosa.transpose %2 {perms = array<i32: 0, 2, 1>} : (tensor<178x1x192xi1>) -> tensor<178x192x1xi1>
    return %0, %3, %5 : tensor<20x13x36x36x37x88xi32>, tensor<39x89x94x90xf32>, tensor<178x192x1xi1>
  }
}
