module {
  func.func @main(%arg0: tensor<63x61x26xi8>, %arg1: tensor<63x26x94xi8>, %arg2: tensor<84x73x88x17x42xi32>, %arg3: tensor<1x73x1x17x42xi32>, %arg4: tensor<35x37xi1>) -> (tensor<35x37xi1>, tensor<84x73x88x17x42xi1>, tensor<63x61x94xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<63x61x26xi8>, tensor<63x26x94xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<63x61x94xi8>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<84x73x88x17x42xi32>, tensor<1x73x1x17x42xi32>) -> tensor<84x73x88x17x42xi32>
    %2 = tosa.logical_not %arg4 : (tensor<35x37xi1>) -> tensor<35x37xi1>
    %3 = tosa.greater %1, %1 : (tensor<84x73x88x17x42xi32>, tensor<84x73x88x17x42xi32>) -> tensor<84x73x88x17x42xi1>
    %4 = tosa.equal %0, %0 : (tensor<63x61x94xi8>, tensor<63x61x94xi8>) -> tensor<63x61x94xi1>
    return %2, %3, %4 : tensor<35x37xi1>, tensor<84x73x88x17x42xi1>, tensor<63x61x94xi1>
  }
}
