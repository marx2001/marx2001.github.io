---
layout: post
title: "Wannier Tools的代码解释"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20251202-code
---



创建虚拟环境
```fortran

! 子程序：计算单个k点的Berry曲率（基于费米能级）
subroutine Berry_curvature_singlek_EF(k, mu, Omega_x, Omega_y, Omega_z)
     !> 计算单个k点的Berry曲率
     !> 费米分布由费米能级iso_energy决定
     !> 参考文献：Physical Review B 74, 195118(2006)
     !> 公式(34)
     !> 作者：Quansheng Wu @ ETHZ, 2018年6月11日
     !> 尝试编写一些简单的子程序
     !> 输出中只有实部有意义
     ! 版权所有 (c) 2018 QuanSheng Wu

     use wmpi        ! 使用MPI并行模块
     use para        ! 使用参数模块
     implicit none   ! 强制显式声明变量

     !> 输入参数，k以倒格矢为单位
     !> mu: 化学势，可通过背栅调节
     real(dp), intent(in) :: k(3), mu
     complex(dp), intent(out) :: Omega_x(Num_wann)  ! x分量Berry曲率
     complex(dp), intent(out) :: Omega_y(Num_wann)  ! y分量Berry曲率
     complex(dp), intent(out) :: Omega_z(Num_wann)  ! z分量Berry曲率

     ! 局部变量声明
     integer :: m, n, i
     real(dp), allocatable :: W(:)                    ! 本征值数组
     real(dp) :: beta_fake                            ! 假想的beta参数（用于费米分布）
     complex(dp), allocatable :: Amat(:, :)           ! 临时矩阵
     complex(dp), allocatable :: DHDk(:, :, :)        ! 哈密顿量对k的导数
     complex(dp), allocatable :: DHDkdag(:, :, :)     ! DHDk的共轭转置
     complex(dp), allocatable :: UU(:, :)             ! 本征向量矩阵
     complex(dp), allocatable :: UU_dag(:, :)         ! UU的共轭转置
     complex(dp), allocatable :: Hamk_bulk(:, :)      ! 体态哈密顿量
     complex(dp), allocatable :: velocity_wann(:, :, :) ! Wannier速度算符
     complex(dp), allocatable :: velocity_Ham(:, :, :) ! 希尔伯特空间速度算符

     !> 费米-狄拉克分布函数
     real(dp), external :: fermi

     ! 分配内存并初始化
     allocate(W(Num_wann))
     allocate(velocity_wann(Num_wann, Num_wann, 3), velocity_Ham(Num_wann, Num_wann, 3))
     allocate(UU(Num_wann, Num_wann), UU_dag(Num_wann, Num_wann), Hamk_bulk(Num_wann, Num_wann))
     allocate(Amat(Num_wann, Num_wann), DHDk(Num_wann, Num_wann, 3), DHDkdag(Num_wann, Num_wann, 3))
     W=0d0; velocity_wann= 0d0; UU= 0d0; UU_dag= 0d0; velocity_Ham= 0d0

     ! 通过HmnR的直接傅里叶变换计算体态哈密顿量
     call ham_bulk_atomicgauge(k, Hamk_bulk)

     !> 使用LAPACK的zheev进行对角化
     UU=Hamk_bulk
     call eigensystem_c( 'V', 'U', Num_wann, UU, W)  ! 'V'=计算本征向量, 'U'=单位矩阵
    !call zhpevx_pack(hamk_bulk,Num_wann, W, UU)   ! 备用对角化方法

     ! 计算本征向量的共轭转置
     UU_dag= conjg(transpose(UU))
     ! 计算原子规范下的速度算符
     call dHdk_atomicgauge(k, velocity_wann)

     !> 将速度算符变换到希尔伯特空间
     do i=1, 3  ! 遍历三个方向(x,y,z)
        ! velocity_wann × UU → Amat
        call mat_mul(Num_wann, velocity_wann(:, :, i), UU, Amat) 
        ! UU_dag × Amat → velocity_Ham
        call mat_mul(Num_wann, UU_dag, Amat, velocity_Ham(:, :, i)) 
     enddo

     ! 初始化Berry曲率为零
     Omega_x=0d0;Omega_y=0d0; Omega_z=0d0
     
     ! 计算Berry曲率的主循环
     do m= 1, Num_wann      ! 遍历所有能带
        do n= 1, Num_wann  ! 遍历所有能带
           ! 跳过简并能带
           if (abs(W(m)-W(n))<eps6) cycle
           
           ! 根据公式计算Berry曲率分量
           ! Ω_x = Σ_{n≠m} [v_H^{y}_{n,m} × v_H^{z}_{m,n}] / (E_m - E_n)^2
           Omega_x(m)= Omega_x(m)+ velocity_Ham(n, m, 2)*velocity_Ham(m, n, 3)/((W(m)-W(n))**2)
           ! Ω_y = Σ_{n≠m} [v_H^{z}_{n,m} × v_H^{x}_{m,n}] / (E_m - E_n)^2
           Omega_y(m)= Omega_y(m)+ velocity_Ham(n, m, 3)*velocity_Ham(m, n, 1)/((W(m)-W(n))**2)
           ! Ω_z = Σ_{n≠m} [v_H^{x}_{n,m} × v_H^{y}_{m,n}] / (E_m - E_n)^2
           Omega_z(m)= Omega_z(m)+ velocity_Ham(n, m, 1)*velocity_Ham(m, n, 2)/((W(m)-W(n))**2)
        enddo ! n循环
     enddo ! m循环

     ! 添加复数因子 -2i
     Omega_x= -Omega_x*2d0*zi
     Omega_y= -Omega_y*2d0*zi
     Omega_z= -Omega_z*2d0*zi

     !> 考虑费米分布的展宽效应
     if (Fermi_broadening<eps6) Fermi_broadening= eps6  ! 防止除零
     beta_fake= 1d0/Fermi_broadening  ! 计算假想温度的倒数
     
     ! 应用费米分布权重
     do m= 1, Num_wann
        Omega_x(m)= Omega_x(m)*fermi(W(m)-mu, beta_fake)  ! f(E_m - μ)
        Omega_y(m)= Omega_y(m)*fermi(W(m)-mu, beta_fake)
        Omega_z(m)= Omega_z(m)*fermi(W(m)-mu, beta_fake)
     enddo

     return
end subroutine Berry_curvature_singlek_EF

! 子程序：计算单个k点的Berry曲率（基于占据能带数，适用于slab系统）
subroutine Berry_curvature_singlek_numoccupied_slab_total(k, Omega_z)
     !> 计算单个k点的Berry曲率
     !> 费米分布由占据能带数决定，而非费米能级
     !> 参考文献：Physical Review B 74, 195118(2006)
     !> 公式(11)，只计算xy分量
     !> 作者：Quansheng Wu, 2018年8月6日
     ! 版权所有 (c) 2018 QuanSheng Wu

     use wmpi
     use para
     implicit none

     !> 输入参数，k以倒格矢为单位（二维k点）
     real(dp), intent(in) :: k(2)
     complex(dp), intent(out) :: Omega_z(1)  ! 总Berry曲率

     ! 局部变量
     integer :: m, n, Mdim, i1, i2
     real(dp), allocatable :: W(:)                    ! 本征值数组
     complex(dp), allocatable :: Amat(:, :)           ! 临时矩阵
     complex(dp), allocatable :: DHDk(:, :, :)        ! 哈密顿量对k的导数
     complex(dp), allocatable :: DHDkdag(:, :, :)     ! DHDk的共轭转置
     complex(dp), allocatable :: UU(:, :)             ! 本征向量矩阵
     complex(dp), allocatable :: UU_dag(:, :)         ! UU的共轭转置
     complex(dp), allocatable :: Hamk_slab(:, :)      ! Slab哈密顿量
     complex(dp), allocatable :: vx(:, :), vy(:, :)   ! x和y方向的速度算符
     complex(dp), allocatable :: Vij_x(:, :, :), Vij_y(:, :, :)  ! 层间跃迁矩阵元

     ! Slab系统的主导哈密顿量维度
     Mdim = Num_wann*Nslab  ! 总维度 = 瓦尼尔能带数 × slab层数

     ! 分配内存
     allocate(W(Mdim))
     allocate(vx(Mdim, Mdim), vy(Mdim, Mdim))
     allocate(UU(Mdim, Mdim), UU_dag(Mdim, Mdim), Hamk_slab(Mdim, Mdim))
     allocate(Amat(Mdim, Mdim), DHDk(Mdim, Mdim, 3), DHDkdag(Mdim, Mdim, 3))
     allocate(Vij_x(-ijmax:ijmax, Num_wann, Num_wann))  ! 层间跃迁矩阵元(x方向)
     allocate(Vij_y(-ijmax:ijmax, Num_wann, Num_wann))  ! 层间跃迁矩阵元(y方向)
     W=0d0; vx= 0d0; vy= 0d0; UU= 0d0; UU_dag= 0d0

     ! 计算层间跃迁速度矩阵元
     call ham_qlayer2qlayer_velocity(k, Vij_x, Vij_y) 
     
     ! 构建完整的slab速度算符矩阵
     do i1=1, nslab  ! 源层
        ! i2目标层索引
        do i2=1, nslab
          ! 只考虑近邻层间耦合
          if (abs(i2-i1).le.ijmax)then
            ! 填充x方向速度算符块
            vx((i2-1)*Num_wann+1:(i2-1)*Num_wann+Num_wann,&
                      (i1-1)*Num_wann+1:(i1-1)*Num_wann+Num_wann )&
            = Vij_x(i2-i1,1:Num_wann,1:Num_wann)
            ! 填充y方向速度算符块
            vy((i2-1)*Num_wann+1:(i2-1)*Num_wann+Num_wann,&
                      (i1-1)*Num_wann+1:(i1-1)*Num_wann+Num_wann )&
            = Vij_y(i2-i1,1:Num_wann,1:Num_wann)
          endif 
        enddo ! i2循环
     enddo ! i1循环

     ! 通过直接傅里叶变换计算slab哈密顿量
     call ham_slab(k, Hamk_slab)

     !> 对角化哈密顿量
     UU=Hamk_slab
     call eigensystem_c( 'V', 'U', Mdim, UU, W)  ! 计算本征值和本征向量

     ! 计算本征向量的共轭转置
     UU_dag= conjg(transpose(UU))
 
     !> 将速度算符变换到希尔伯特空间
     call mat_mul(Mdim, vx, UU, Amat) 
     call mat_mul(Mdim, UU_dag, Amat, vx)   ! vx在能量本征基下表示
     call mat_mul(Mdim, vy, UU, Amat) 
     call mat_mul(Mdim, UU_dag, Amat, vy)   ! vy在能量本征基下表示

     ! 初始化Berry曲率
     Omega_z=0d0
     
     ! 计算总Berry曲率：只对占据态到未占据态求和
     do m= 1, NumOccupied*Nslab        ! 占据态（价带）
        do n= NumOccupied*Nslab+1, Mdim ! 未占据态（导带）
           ! Ω_z = Σ_{m∈occ, n∈unocc} [v_x^{n,m} × v_y^{m,n}] / (E_m - E_n)^2
           Omega_z(1)= Omega_z(1)+ vx(n, m)*vy(m, n)/((W(m)-W(n))**2)
        enddo ! m循环
     enddo ! n循环

     ! 添加复数因子并转换单位
     Omega_z= -aimag(Omega_z*2d0)/Angstrom2atomic**2  ! 取虚部并转换单位

     return
end subroutine Berry_curvature_singlek_numoccupied_slab_total

! 其他子程序的注释类似，主要区别在于：
! 1. Berry_curvature_singlek_numoccupied_total: 计算总Berry曲率（基于占据能带数）
! 2. Berry_curvature_singlek_numoccupied: 计算每个能带的Berry曲率（基于占据能带数）
! 3. Berry_curvature_singlek_allbands: 使用Kubo公式计算所有能带的Berry曲率
! 4. orbital_magnetization_singlek_allbands: 计算轨道磁化强度
! 5. get_Vmn_Ham_nondiag: 获取非对角速度矩阵元
! 6. get_Dmn_Ham: 定义D矩阵（Berry连接）
! 7. Berry_curvature_slab: 计算slab系统的Berry曲率并输出结果

! 主程序部分：计算slab系统的Berry曲率
subroutine Berry_curvature_slab
     !> 计算Berry曲率
     !> 参考文献：Physical Review B 74, 195118(2006)
     !> 作者：Quansheng Wu @ EPFL, 2018年8月6日
     ! 版权所有 (c) 2018 QuanSheng Wu

     use wmpi
     use para
     implicit none
    
     ! MPI和循环变量
     integer :: ik, ierr, Mdim, i, j

     ! k点坐标
     real(dp) :: k(2)  

     !> k点网格
     real(dp), allocatable :: k12(:, :)       ! 分数坐标k点
     real(dp), allocatable :: k12_xyz(:, :)   ! 笛卡尔坐标k点
   
     real(dp), external :: norm  ! 范数函数

     !> Berry曲率数组
     complex(dp), allocatable :: Omega_z(:)    ! 每个k点的Berry曲率
     complex(dp), allocatable :: Omega(:)      ! 每个k点的总Berry曲率
     complex(dp), allocatable :: Omega_mpi(:)  ! MPI归约后的Berry曲率

     ! Slab系统维度
     Mdim = Num_wann*Nslab

     ! 分配内存
     allocate( k12(2, Nk1*Nk2))           ! 2维k点网格（分数坐标）
     allocate( k12_xyz(2, Nk1*Nk2))       ! 笛卡尔坐标k点
     allocate( Omega_z(Mdim))             ! Berry曲率数组
     allocate( Omega    (Nk1*Nk2))        ! 总Berry曲率
     allocate( Omega_mpi(Nk1*Nk2))        ! MPI归约结果
     k12=0d0
     k12_xyz=0d0
     omega= 0d0
     omega_mpi= 0d0
    
     !> 构建k点网格，以K2d_start为中心
     ik=0
     do i= 1, nk1  ! k1方向循环
        do j= 1, nk2  ! k2方向循环
           ik=ik+1
           ! 构建分数坐标k点网格
           k12(:, ik)=K2D_start+ (i-1)*K2D_vec1/dble(nk1-1) &
                      + (j-1)*K2D_vec2/dble(nk2-1)- (K2D_vec1+K2D_vec2)/2d0
           ! 转换为笛卡尔坐标
           k12_xyz(:, ik)= k12(1, ik)* Ka2+ k12(2, ik)* Kb2
        enddo
     enddo

     ! 并行计算每个k点的Berry曲率
     do ik= 1+ cpuid, Nk1*Nk2, num_cpu  ! CPU负载均衡
        if (cpuid==0) write(stdout, *)'Berry curvature ik, nk1*nk2 ', ik, Nk1*Nk2

        !> 对角化哈密顿量
        k= k12(:, ik)  ! 当前k点

        Omega_z= 0d0   ! 初始化

        ! 计算单个k点的Berry曲率
        call Berry_curvature_singlek_numoccupied_slab_total(k, Omega_z(1))
        ! 对所有能带求和得到总Berry曲率
        Omega(ik) = sum(Omega_z)

     enddo ! ik循环

     ! 初始化MPI归约数组
     Omega_mpi= 0d0

#if defined (MPI)
     ! MPI归约：将所有处理器的结果求和
     call mpi_allreduce(Omega,Omega_mpi,size(Omega_mpi),&
                       mpi_dc,mpi_sum,mpi_cmw,ierr)
#else
     ! 串行版本：直接复制
     Omega_mpi= Omega
#endif

     !> 输出Berry曲率到文件
     outfileindex= outfileindex+ 1  ! 更新输出文件索引
     if (cpuid==0) then  ! 只在主进程输出
        open(unit=outfileindex, file='Berrycurvature_slab.dat')
        write(outfileindex, '(20a28)')'# Unit of Berry curvature is Angstrom^2'
        write(outfileindex, '(20a28)')'# kx (1/A)', 'ky (1/A)', &
           'Omega_z'
        ik= 0
        do i= 1, nk1
           do j= 1, nk2
              ik= ik+ 1
              ! 写入k点和Berry曲率（转换单位）
              write(outfileindex, '(20E28.10)')k12_xyz(:, ik)*Angstrom2atomic, &
                   real(Omega_mpi(ik))/Angstrom2atomic**2
           enddo
           write(outfileindex, *) ' '  ! 空行分隔
        enddo

        close(outfileindex)
! 结束之前的if语句块
     endif

     !> 生成gnuplot脚本来绘制Berry曲率图
     outfileindex= outfileindex+ 1  ! 增加输出文件索引
     if (cpuid==0) then  ! 只在主进程执行

        ! 打开gnuplot脚本文件
        open(unit=outfileindex, file='Berrycurvature_slab.gnu')
        ! 设置字符编码
        write(outfileindex, '(a)')"set encoding iso_8859_1"
        ! 设置终端为png格式，高分辨率
        write(outfileindex, '(a)')'#set terminal  pngcairo  truecolor enhanced size 1920, 1680 font ",60"'
        write(outfileindex, '(a)')'set terminal  png       truecolor enhanced size 1920, 1680 font ",60"'
        ! 设置输出文件名
        write(outfileindex, '(a)')"set output 'Berrycurvature_slab.png'"
        ! 设置调色板
        write(outfileindex, '(a)')"set palette rgbformulae 33,13,10"
        ! 取消z轴刻度
        write(outfileindex, '(a)')"unset ztics"
        ! 设置纵横比
        write(outfileindex, '(a)')"set size ratio -1"
        ! 设置图形大小
        write(outfileindex, '(a)')"set size 0.9, 0.95"
        ! 设置图形原点
        write(outfileindex, '(a)')"set origin 0.05, 0.02"
        ! 取消图例
        write(outfileindex, '(a)')"unset key"
        ! 启用pm3d绘图模式
        write(outfileindex, '(a)')"set pm3d"
        ! 注释掉的z轴和颜色条范围设置
        write(outfileindex, '(a)')"#set zbrange [ -10: 10] "
        write(outfileindex, '(a)')"#set cbrange [ -100: 100] "
        ! 设置视图为映射模式
        write(outfileindex, '(a)')"set view map"
        ! 设置边框线宽
        write(outfileindex, '(a)')"set border lw 3"
        ! 设置x轴标签
        write(outfileindex, '(a)')"set xlabel 'k (1/{\305})'"
        ! 设置y轴标签
        write(outfileindex, '(a)')"set ylabel 'k (1/{\305})'"
        ! 设置x轴范围自动
        write(outfileindex, '(a)')"set xrange [] noextend"
        ! 设置y轴范围自动
        write(outfileindex, '(a)')"set yrange [] noextend"
        ! 设置y轴刻度和样式
        write(outfileindex, '(a)')"set ytics 0.5 nomirror scale 0.5"
        ! 设置pm3d插值
        write(outfileindex, '(a)')"set pm3d interpolate 2,2"
        ! 设置标题
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_z (Ang^2)'"
        ! 显示颜色条
        write(outfileindex, '(a)')"set colorbox"
        ! 绘制3D彩色图
        write(outfileindex, '(a)')"splot 'Berrycurvature_slab.dat' u 1:2:3 w pm3d"
        ! 关闭文件
        close(outfileindex)
     endif


#if defined (MPI)
     ! MPI同步屏障
     call mpi_barrier(mpi_cmw, ierr)
#endif

     ! 释放内存
     deallocate( k12)
     deallocate( k12_xyz)
     deallocate( Omega_z)
     deallocate( Omega, Omega_mpi)
 
     return
  end subroutine Berry_curvature_slab

  ! 子程序：计算k路径上的Berry曲率（基于占据能带）
  subroutine Berry_curvature_line_occupied
     !> 计算由KPATH_BULK定义的k线上的Berry曲率
     !> 参考文献：Physical Review B 74, 195118(2006)
     !> 公式(9), 公式(34)
     !> 作者：Quansheng Wu @ EPFL, 2018年9月28日
     ! 版权所有 (c) 2018 QuanSheng Wu

     use wmpi
     use para
     implicit none
    
     ! 局部变量
     integer :: ik, ierr, i
     real(dp) :: k(3), o1(3), k_cart(3)  ! k点坐标
     real(dp) :: time_start, time_end, time_start0, ybound_min, ybound_max  ! 时间统计和边界
     real(dp), external :: norm  ! 范数函数

     ! 分配内存：Berry曲率数组（每个能带）
     complex(dp), allocatable :: Omega_x(:), Omega_y(:), Omega_z(:)
     !> Berry曲率数组 (3个分量, k点)
     real(dp), allocatable :: Omega(:, :), Omega_mpi(:, :)
     !> Berry曲率数组 (3个分量, 能带, k点)
     real(dp), allocatable :: Omega_sep_bk(:, :), Omega_sep_bk_mpi(:, :)
     !> 能量本征值
     real(dp), allocatable :: eigv(:,:)
     real(dp), allocatable :: eigv_mpi(:,:)

     ! 分配内存空间
     allocate( Omega_x(Num_wann))
     allocate( Omega_y(Num_wann))
     allocate( Omega_z(Num_wann))
     allocate( eigv    (Num_wann, nk3_band))
     allocate( eigv_mpi(Num_wann, nk3_band))
     allocate( Omega    (3, nk3_band))
     allocate( Omega_mpi(3, nk3_band))
     omega= 0d0
     omega_mpi= 0d0
    
     ! 初始化时间统计
     time_start= 0d0
     time_start0= 0d0
     call now(time_start0)  ! 获取开始时间
     time_start= time_start0
     time_end  = time_start0
     
     ! 主循环：遍历k点（并行）
     do ik= 1+ cpuid, nk3_band, num_cpu  ! CPU负载均衡
        ! 进度输出
        if (cpuid==0.and. mod(ik/num_cpu, 100)==0) &
           write(stdout, '(a, i9, "  /", i10, a, f10.1, "s", a, f10.1, "s")') &
           ' Berry curvature: ik', ik, nk3_band, ' time left', &
           (nk3_band-ik)*(time_end- time_start)/num_cpu, &  ! 预估剩余时间
           ' time elapsed: ', time_end-time_start0            ! 已用时间

        !> 分数坐标下的k点
        k= kpath_3d(:, ik)

        call now(time_start)  ! 记录开始时间
 
        ! 初始化Berry曲率
        Omega_x= 0d0
        Omega_y= 0d0
        Omega_z= 0d0

        ! 根据不同的计算方法选择不同的子程序
        if (Berrycurvature_kpath_EF_calc) then
           ! 基于费米能级的计算
           call Berry_curvature_singlek_EF(k, iso_energy, Omega_x, Omega_y, Omega_z)
        else if (BerryCurvature_kpath_Occupied_calc) then
           ! 基于占据能带的计算（只返回总和）
           call Berry_curvature_singlek_numoccupied_total(k, Omega_x(1), Omega_y(1), Omega_z(1))
        else
           ! 错误处理
           write(*, *) 'ERROR: In subroutine Berry_curvature_line, we only support BerryCurvature_kpath_Occupied_calc '
           write(*, *) ' and Berrycurvature_kpath_EF_calc '
           stop
        endif
 
        ! 对能带求和得到总Berry曲率
        Omega(1, ik) = real(sum(Omega_x))
        Omega(2, ik) = real(sum(Omega_y))
        Omega(3, ik) = real(sum(Omega_z))
        call now(time_end)  ! 记录结束时间
     enddo ! ik循环

     ! 初始化MPI归约数组
     Omega_mpi= 0d0

#if defined (MPI)
     ! MPI归约：求和所有处理器的计算结果
     call mpi_allreduce(Omega,Omega_mpi,size(Omega_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
#else
     ! 串行版本：直接复制
     Omega_mpi= Omega
#endif

     !> 输出Berry曲率数据到文件
     outfileindex= outfileindex+ 1
     if (cpuid==0) then  ! 主进程输出
        open(unit=outfileindex, file='Berrycurvature_line.dat')
        ! 写入文件头信息
        write(outfileindex, '(20a18)')'# Column 1 kpath with accumulated length in the kpath'
        write(outfileindex, '(20a18)')'# Column 2-4 Berry curvature (Ang^2)'
        write(outfileindex, '(20a18)')'# k (1/A)', &
           'real(Omega_x)', 'real(Omega_y)', 'real(Omega_z)'

        ! 写入数据
        do ik= 1, nk3_band
           k=kpath_3d(:, ik)
           ! 写入累积k路径长度和Berry曲率分量（转换单位）
           write(outfileindex, '(20E18.8)')k3len(ik)*Angstrom2atomic, real(Omega_mpi(:, ik))/Angstrom2atomic**2
        enddo

        close(outfileindex)
     endif

     ! 计算y轴显示范围
     ybound_min= minval(real(Omega_mpi))-2
     ybound_max= maxval(real(Omega_mpi))+5 
     
     !> 生成gnuplot脚本
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature_line.gnu')
        write(outfileindex, '(a)')"set encoding iso_8859_1"
        ! 设置PDF输出
        write(outfileindex, '(a)') 'set terminal pdf enhanced color font ",16"'
        write(outfileindex, '(a)')"set output 'Berrycurvature_line.pdf'"
        ! 设置绘图样式
        write(outfileindex, '(a)')'set style data linespoints'
        write(outfileindex, '(a)')'unset key'  ! 取消图例
        write(outfileindex, '(a)')'set pointsize 0.8'
        write(outfileindex, '(a)')'set view 0,0'  ! 2D视图
        ! 设置x轴范围
        write(outfileindex, 202, advance="no") (k3line_name(i), k3line_stop(i)*Angstrom2atomic, i=1, nk3lines)
        write(outfileindex, 203)k3line_name(nk3lines+1), k3line_stop(nk3lines+1)*Angstrom2atomic
  
        ! 绘制高亮线和垂直标记
        do i=1, nk3lines-1
           if (index(Particle,'phonon')/=0) then
              ! 声子模式的范围设置
              write(outfileindex, 204)k3line_stop(i+1)*Angstrom2atomic, 0.0, k3line_stop(i+1)*Angstrom2atomic, ybound_max
           else
              ! 电子模式的范围设置
              write(outfileindex, 204)k3line_stop(i+1)*Angstrom2atomic, ybound_min, k3line_stop(i+1)*Angstrom2atomic, ybound_max
           endif
        enddo
        
        ! 绘制三条Berry曲率分量曲线
        write(outfileindex, '(a)')"plot 'Berrycurvature_line.dat' \"  
        write(outfileindex, '(a)')"u 1:2 w lp lc rgb 'red'   lw 2 pt 7 ps 0.2 title '{/Symbol W}_x', \" 
        write(outfileindex, '(a)')"'' u 1:3 w lp lc rgb 'green' lw 2 pt 7 ps 0.2 title '{/Symbol W}_y', \" 
        write(outfileindex, '(a)')"'' u 1:4 w lp lc rgb 'blue'  lw 2 pt 7 ps 0.2 title '{/Symbol W}_z' "
        close(outfileindex)
     endif

202 format('set xtics (',20('"',A3,'" ',F10.5,','))  ! x轴刻度格式
203 format(A3,'" ',F10.5,')')                        ! x轴刻度结束格式
204 format('set arrow from ',F10.5,',',F10.5,' to ',F10.5,',',F10.5, ' nohead')  ! 箭头格式


#if defined (MPI)
     ! MPI同步
     call mpi_barrier(mpi_cmw, ierr)
#endif

     ! 释放内存
     deallocate( Omega_x, Omega_y, Omega_z)
     deallocate( Omega, Omega_mpi)
 
     return
  end subroutine Berry_curvature_line_occupied

  ! 子程序：计算立方体格点中的Berry曲率
  subroutine berry_curvature_cube
     !> 计算在KCUBE_BULK定义的立方体内的Berry曲率
     !> 参考文献：Physical Review B 74, 195118(2006)
     !> 公式(34)
     !> 作者：Quansheng Wu @ EPFL, 2018年9月28日
     ! 版权所有 (c) 2018 QuanSheng Wu

     use wmpi
     use para
     implicit none
    
     ! 局部变量
     integer :: ik, ierr, ikx, iky, ikz, n_kpoints, i, m, n
     real(dp) :: k(3), o1(3), k_cart(3), emin, emax
     real(dp) :: time_start, time_end, time_start0
     real(dp), external :: norm

     !> Berry曲率和轨道磁化强度数组 (能带, 3分量, k点)
     real(dp), allocatable :: Omega_allk(:, :, :), Omega_allk_mpi(:, :, :)
     real(dp), allocatable :: m_OrbMag_allk(:, :, :), m_OrbMag_allk_mpi(:, :, :)
     real(dp), allocatable :: Omega_BerryCurv(:, :), m_OrbMag(:, :)

     !> Wannier基下的速度矩阵
     complex(dp), allocatable :: Vmn_wann(:, :, :)

     !> 哈密顿量基下的速度矩阵
     complex(dp), allocatable :: Vmn_Ham(:, :, :), Vmn_Ham_nondiag(:, :, :)
     complex(dp), allocatable :: Dmn_Ham(:, :, :)  ! Berry连接矩阵

     !> 哈密顿量、本征值和本征向量
     complex(dp), allocatable :: UU(:, :)
     real(dp), allocatable :: W(:)
     real(dp), allocatable :: eigval_allk(:, :), eigval_allk_mpi(:, :)

     ! 根据计算模式确定k点总数
     if (BerryCurvature_Cube_calc) then
        n_kpoints=Nk1*Nk2*Nk3  ! 立方体网格
     else
        n_kpoints= nk3_band    ! k路径模式
     endif

     ! 分配内存
     allocate(Vmn_wann(Num_wann, Num_wann, 3), Vmn_Ham(Num_wann, Num_wann, 3))
     allocate(Dmn_Ham(Num_wann,Num_wann,3), Vmn_Ham_nondiag(Num_wann, Num_wann, 3))
     allocate(W(Num_wann))
     allocate(UU(Num_wann, Num_wann))
     allocate(eigval_allk(Num_wann, n_kpoints))
     allocate(eigval_allk_mpi(Num_wann, n_kpoints))
     allocate( Omega_BerryCurv(Num_wann, 3), m_OrbMag(Num_wann, 3))
     allocate( Omega_allk    (Num_wann, 3, n_kpoints))
     allocate( Omega_allk_mpi(Num_wann, 3, n_kpoints))
     allocate( m_OrbMag_allk    (Num_wann, 3, n_kpoints))
     allocate( m_OrbMag_allk_mpi(Num_wann, 3, n_kpoints))
     
     ! 初始化数组
     Omega_BerryCurv= 0d0
     m_OrbMag=0d0
     Omega_allk= 0d0
     eigval_allk= 0d0
     m_OrbMag_allk=0d0
     Omega_allk_mpi= 0d0
     eigval_allk_mpi= 0d0
     m_OrbMag_allk_mpi=0d0
    
     ! 时间统计初始化
     time_start= 0d0
     time_start0= 0d0
     call now(time_start0)
     time_start= time_start0
     time_end  = time_start0
     
     ! 主循环：遍历所有k点
     do ik= 1+ cpuid, n_kpoints, num_cpu
        ! 进度输出
        if (cpuid==0.and. mod(ik/num_cpu, 100)==0) &
           write(stdout, '(a, i9, "  /", i10, a, f10.1, "s", a, f10.1, "s")') &
           ' Berry curvature: ik', ik, n_kpoints, ' time left', &
           (n_kpoints-ik)*(time_end- time_start)/num_cpu, &
           ' time elapsed: ', time_end-time_start0 

        !> 根据计算模式生成k点
        if (BerryCurvature_Cube_calc) then
           !> 体布里渊区模式：从三维网格索引计算k点坐标
           ikx= (ik-1)/(nk2*nk3)+1  ! x方向索引
           iky= ((ik-1-(ikx-1)*Nk2*Nk3)/nk3)+1  ! y方向索引
           ikz= (ik-(iky-1)*Nk3- (ikx-1)*Nk2*Nk3)  ! z方向索引
           ! 计算分数坐标k点
           k= K3D_start_cube+ K3D_vec1_cube*(ikx-1)/dble(nk1)  &
              + K3D_vec2_cube*(iky-1)/dble(nk2)  &
              + K3D_vec3_cube*(ikz-1)/dble(nk3) 
 
        elseif (BerryCurvature_kpath_sepband_calc) then
           !> k路径模式：直接从预定义路径获取k点
           k= kpath_3d(:, ik)
        endif

        call now(time_start)  ! 记录开始时间
        
        !> 通过HmnR的直接傅里叶变换计算体态哈密顿量
        call ham_bulk_atomicgauge(k, UU)
   
        !> 对角化哈密顿量
        call eigensystem_c( 'V', 'U', Num_wann, UU, W)
        eigval_allk(:, ik) = W  ! 保存本征值

        !> 在哈密顿量基下获取速度算符
        call dHdk_atomicgauge_Ham(k, UU, Vmn_Ham)

        ! 计算Berry连接矩阵和非对角速度矩阵元
        call get_Dmn_Ham(W, Vmn_Ham, Dmn_Ham)
        call get_Vmn_Ham_nondiag(Vmn_Ham, Vmn_Ham_nondiag)

        ! 计算Berry曲率和轨道磁化强度
        call Berry_curvature_singlek_allbands(Dmn_Ham, Omega_BerryCurv)
        call orbital_magnetization_singlek_allbands(Dmn_Ham, Vmn_Ham_nondiag, m_OrbMag)
        
        ! 保存结果
        Omega_allk(:, :, ik) = Omega_BerryCurv
        m_OrbMag_allk(:, :, ik) = m_OrbMag

        call now(time_end)  ! 记录结束时间
     enddo ! ik循环

#if defined (MPI)
     ! MPI归约所有结果
     call mpi_allreduce(Omega_allk,Omega_allk_mpi,size(Omega_allk_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
     call mpi_allreduce(eigval_allk,eigval_allk_mpi,size(eigval_allk_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
     call mpi_allreduce(m_OrbMag_allk,m_OrbMag_allk_mpi,size(m_OrbMag_allk_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
#else
     ! 串行版本
     Omega_allk_mpi= Omega_allk
     eigval_allk_mpi= eigval_allk
     m_OrbMag_allk_mpi= m_OrbMag_allk
#endif

     ! 立方体计算模式的输出部分
     IF (BerryCurvature_Cube_calc) THEN
     !> 输出Berry曲率和轨道磁化强度到Fermisurfer软件可读的文件
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature_norm.frmsf')
        ! 写入网格尺寸
        write(outfileindex, *) Nk1, Nk2, Nk3
        write(outfileindex, *) 1  ! 自旋极化数
        write(outfileindex, *) Num_wann  ! 能带数
        ! 写入晶格矢量
        write(outfileindex, '(3f12.6)') Origin_cell%Kua
        write(outfileindex, '(3f12.6)') Origin_cell%Kub
        write(outfileindex, '(3f12.6)') Origin_cell%Kuc
        ! 写入能量本征值（相对于费米能级）
        do m=1, Num_wann
           do ik= 1, n_kpoints
              write(outfileindex, '(E18.10)') eigval_allk_mpi(m, ik)-iso_energy
           enddo
        enddo
        ! 写入Berry曲率模长
        do m=1, Num_wann
           do ik= 1, n_kpoints
              o1= Omega_allk_mpi(m, :, ik)/Angstrom2atomic**2
              write(outfileindex, '(E18.10)') norm(o1)
           enddo
        enddo
        close(outfileindex)
     endif

     ! 类似地输出x分量Berry曲率...
     ! [其余输出文件的代码类似，分别输出不同分量和轨道磁化强度]

     !> 轨道磁化强度输出（x分量）
     outfileindex= outfileindex+ 1  ! 增加输出文件索引
     if (cpuid==0) then  ! 只在主进程执行
        ! 打开轨道磁化强度x分量文件（Fermisurfer格式）
        open(unit=outfileindex, file='Orbital_magnetization_x.frmsf')
        ! 写入网格尺寸
        write(outfileindex, *) Nk1, Nk2, Nk3
        write(outfileindex, *) 1  ! 自旋极化数
        write(outfileindex, *) Num_wann  ! 能带数
        ! 写入倒格矢（晶格矢量）
        write(outfileindex, '(3f12.6)') Origin_cell%Kua
        write(outfileindex, '(3f12.6)') Origin_cell%Kub
        write(outfileindex, '(3f12.6)') Origin_cell%Kuc
        ! 写入能量本征值（相对于费米能级）
        do m=1, Num_wann
           do ik= 1, n_kpoints
              write(outfileindex, '(E18.10)') eigval_allk_mpi(m, ik)-iso_energy
           enddo
        enddo
        ! 写入轨道磁化强度x分量
        do m=1, Num_wann
           do ik= 1, n_kpoints
              o1= m_OrbMag_allk_mpi(m, :, ik)  ! 当前能带和k点的轨道磁化强度矢量
              write(outfileindex, '(E18.10)') o1(1)  ! 只写入x分量
           enddo
        enddo
        close(outfileindex)  ! 关闭文件
     endif

     ! 条件编译：如果不是立方体计算模式
     ELSE

     !> 输出Berry曲率数据到文本文件（分波带模式）
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature_line_sepband.txt')
        ! 写入文件头信息
        write(outfileindex, '(a)')'# Column 1 kpath with accumulated length in the kpath, Coloum 2: energy'
        write(outfileindex, '(a)')'# Column 2-4 Berry curvature (Ang^2)'
        ! 写入列号信息
        write(outfileindex, "('#column', i5, 20i16)")(i, i=1, 8)
        ! 写入详细的列描述
        write(outfileindex, '(20a16)')'# k (1/A)', " eig", &
           'Omega_x', 'Omega_y', 'Omega_z', &  ! Berry曲率三个分量
           'm_x', 'm_y', 'm_z'               ! 轨道磁化强度三个分量
        ! 遍历所有能带和k点写入数据
        do i=1, Num_wann  ! 遍历能带
           do ik=1, n_kpoints  ! 遍历k点
              ! 写入k路径长度、能量、Berry曲率、轨道磁化强度
              write(outfileindex, '(200E16.6)')k3len(ik)*Angstrom2atomic, &  ! k路径累积长度
                 eigval_allk_mpi(i, ik)/eV2Hartree, &  ! 能量（转换为eV）
                 Omega_allk_mpi(i, :, ik), &           ! Berry曲率矢量
                 m_OrbMag_allk_mpi(i, :, ik)          ! 轨道磁化强度矢量
           enddo
           write(outfileindex, *)' '  ! 每个能带后空行分隔
        enddo
        close(outfileindex)
     endif

     ! 注意：这里有个重复的close语句，可能是代码错误
     close(outfileindex)
     endif

     !> 计算能量带的最小值和最大值
     emin=  minval(eigval_allk_mpi)/eV2Hartree-0.5d0  ! 最小值减去偏移量
     emax=  maxval(eigval_allk_mpi)/eV2Hartree+0.5d0  ! 最大值加上偏移量

      !> 生成gnuplot脚本用于绘制分波带Berry曲率图
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature_line_sepband.gnu')
        ! 写入gnuplot脚本头部信息
        write(outfileindex, '(a)') '# requirement: gnuplot version>5.4'
        write(outfileindex, '(2a)') '# Please open the data file to check the data: Berrycurvature_line_sepband.txt  '
        ! 设置终端和输出
        write(outfileindex, '(a)') 'set terminal pdf enhanced color font ",24"'
        ! 设置调色板
        write(outfileindex,'(2a)') 'set palette defined ( 0  "green", ', &
           '5 "yellow", 10 "red" )'
        write(outfileindex, '(3a)')"set output 'Berrycurvature_line_sepband.pdf'"
        ! 设置绘图样式
        write(outfileindex, '(a)')'set style data linespoints'
        write(outfileindex, '(a)')'unset key'  ! 取消图例
        write(outfileindex, '(a)')'set pointsize 0.8'
        write(outfileindex, '(a)')'#set xtics font ",24"'
        write(outfileindex, '(a)')'#set ytics font ",24"'
        write(outfileindex, '(a)')'#set ylabel font ",24"'
        write(outfileindex, '(a)')'set ylabel offset 0.5,0'
        write(outfileindex, '(a)')'set border lw 2'
        ! 设置x轴范围
        write(outfileindex, '(a, f10.5, a)')'set xrange [0: ', maxval(k3len*Angstrom2atomic), ']'
        ! 定义能量范围变量
        write(outfileindex, '(a,f12.6)')'emin=', emin
        write(outfileindex, '(a,f12.6)')'emax=', emax
        write(outfileindex, '(a)')' '
        ! 设置颜色条范围
        write(outfileindex, '(a)')'#set cbrange if the number is too large meeting some band crossings'
        write(outfileindex, '(a)')'set cbrange [-100:100]'
        write(outfileindex, '(a)')' '
        ! 根据粒子类型设置y轴
        if (index(Particle,'phonon')/=0) then
           ! 声子模式：频率范围
           write(outfileindex, '(a)')'set yrange [0: emax]'
           write(outfileindex, '(a)')'set ylabel "Frequency (THz)"'
        else
           ! 电子模式：能量范围
           write(outfileindex, '(a)')'set ylabel "Energy (eV)"'
           write(outfileindex, '(a)')'set yrange [ emin : emax ]'
        endif
        ! 写入x轴刻度和垂直线标记
        write(outfileindex, 202, advance="no") (k3line_name(i), k3line_stop(i)*Angstrom2atomic, i=1, Nk3lines)
        write(outfileindex, 203)k3line_name(Nk3lines+1), k3line_stop(Nk3lines+1)*Angstrom2atomic
  
        ! 绘制高亮线和垂直标记
        do i=1, Nk3lines-1
           if (index(Particle,'phonon')/=0) then
              ! 声子模式的箭头设置
              write(outfileindex, 204)k3line_stop(i+1)*Angstrom2atomic, '0.0', k3line_stop(i+1)*Angstrom2atomic, 'emax'
           else
              ! 电子模式的箭头设置
              write(outfileindex, 204)k3line_stop(i+1)*Angstrom2atomic, 'emin', k3line_stop(i+1)*Angstrom2atomic, 'emax'
           endif
        enddo
        ! 绘制彩色线条图
        write(outfileindex, '(4a)')"plot 'Berrycurvature_line_sepband.txt' u 1:2:5 ",  &
           " w lp lw 2 pt 7  ps 0.2 lc palette, 0 w l lw 2 dt 2"
        close(outfileindex)
     endif

202 format('set xtics (',20('"',A3,'" ',F10.5,','))  ! x轴刻度格式定义
203 format(A3,'" ',F10.5,')')                        ! x轴刻度结束格式
204 format('set arrow from ',F10.5,',',A5,' to ',F10.5,',',A5, ' nohead lw 2')  ! 箭头格式定义

     ! 结束IF (BerryCurvature_Cube_calc)块
     ENDIF

#if defined (MPI)
     ! MPI进程同步屏障
     call mpi_barrier(mpi_cmw, ierr)
#endif

     ! 子程序返回
     return
  end subroutine berry_curvature_cube

  ! 子程序：计算完整平面上的Berry曲率和轨道磁化强度
  subroutine Berry_curvature_plane_full
     !> 为选定能带计算Berry曲率和轨道磁化强度
     !> 参考文献：Physical Review B 74, 195118(2006) 公式(34)
     !> 作者：Quansheng Wu @ EPFL, 2020年8月20日
     ! 版权所有 (c) 2020 QuanSheng Wu

     use wmpi
     use para
     implicit none
    
     ! 局部变量声明
     integer :: ik, i, j, m, n, ierr, nkmesh2
     real(dp) :: k(3), o1(3), vmin, vmax, kxy_plane(3)  ! k点和辅助变量
     !> k点切片
     real(dp), allocatable :: kslice(:, :), kslice_xyz(:, :)  ! k点网格
     real(dp) :: time_start, time_end, time_start0  ! 时间统计
     real(dp), allocatable :: W(:)  ! 本征值数组
     !> Berry曲率和轨道磁化强度数组 (类型, 3分量, k点)
     real(dp), allocatable :: Omega_allk_Occ(:, :, :), Omega_allk_Occ_mpi(:, :, :)  ! 基于占据数的Berry曲率
     real(dp), allocatable :: m_OrbMag_allk_Occ(:, :, :), m_OrbMag_allk_Occ_mpi(:, :, :)  ! 基于占据数的轨道磁化
     real(dp), allocatable :: Omega_allk_EF(:, :, :), Omega_allk_EF_mpi(:, :, :)    ! 基于费米能级的Berry曲率
     real(dp), allocatable :: m_OrbMag_allk_EF(:, :, :), m_OrbMag_allk_EF_mpi(:, :, :)  ! 基于费米能级的轨道磁化
     real(dp), allocatable :: Omega_BerryCurv(:, :), m_OrbMag(:, :)  ! 单个k点的结果
     real(dp) :: beta_fake, fermi, delta, norm  ! 物理常数和函数

     ! 初始化网格大小和数组
     nkmesh2= Nk1*Nk2  ! 二维k点网格总数
     ! 分配内存空间
     allocate( Omega_BerryCurv(Num_wann, 3), m_OrbMag(Num_wann, 3))
     allocate( Omega_allk_Occ    (2, 3, nkmesh2))  ! 2种求和方式×3分量×k点数
     allocate( Omega_allk_Occ_mpi(2, 3, nkmesh2))
     allocate( m_OrbMag_allk_Occ    (2, 3, nkmesh2))
     allocate( m_OrbMag_allk_Occ_mpi(2, 3, nkmesh2))
     allocate( Omega_allk_EF    (2, 3, nkmesh2))   ! 费米能级方法
     allocate( Omega_allk_EF_mpi(2, 3, nkmesh2))
     allocate( m_OrbMag_allk_EF    (2, 3, nkmesh2))
     allocate( m_OrbMag_allk_EF_mpi(2, 3, nkmesh2))
     allocate( kslice(3, nkmesh2))                ! k点网格（分数坐标）
     allocate( kslice_xyz(3, nkmesh2))            ! k点网格（笛卡尔坐标）
     allocate( W(Num_wann))                      ! 本征值数组
     
     ! 初始化数组为零
     W=0d0
     m_OrbMag=0d0
     Omega_BerryCurv= 0d0
     Omega_allk_Occ= 0d0
     m_OrbMag_allk_Occ=0d0
     Omega_allk_Occ_mpi= 0d0
     m_OrbMag_allk_Occ_mpi=0d0
     Omega_allk_EF= 0d0
     m_OrbMag_allk_EF=0d0
     Omega_allk_EF_mpi= 0d0
     m_OrbMag_allk_EF_mpi=0d0
     kslice=0d0
     kslice_xyz=0d0
    
     !> 构建k点切片，以K3d_start为中心
     ik =0
     do i= 1, nk1  ! k1方向循环
        do j= 1, nk2  ! k2方向循环
           ik =ik +1
           ! 构建分数坐标k点网格
           kslice(:, ik)= K3D_start+ K3D_vec1*(i-1)/dble(nk1-1)  &
                     + K3D_vec2*(j-1)/dble(nk2-1) - (K3D_vec1+ K3D_vec2)/2d0
           ! 转换为笛卡尔坐标
           kslice_xyz(:, ik)= kslice(1, ik)* Origin_cell%Kua+ kslice(2, ik)* Origin_cell%Kub+ kslice(3, ik)* Origin_cell%Kuc 
        enddo
     enddo

     ! 时间统计初始化
     time_start= 0d0
     time_start0= 0d0
     call now(time_start0)  ! 获取开始时间
     time_start= time_start0
     time_end  = time_start0
     
     ! 主循环：遍历二维k点网格（并行）
     do ik= 1+ cpuid, nkmesh2, num_cpu
        ! 进度输出
        if (cpuid==0.and. mod((ik-1)/num_cpu, 100)==0) &
           write(stdout, '(a, i9, "  /", i10, a, f10.1, "s", a, f10.1, "s")') &
           ' Berry curvature: ik', ik, nkmesh2, ' time left', &
           (nkmesh2-ik)*(time_end- time_start)/num_cpu, &  ! 预估剩余时间
           ' time elapsed: ', time_end-time_start0            ! 已用时间

        call now(time_start)  ! 记录开始时间
 
        !> 对角化哈密顿量
        k= kslice(:, ik)  ! 当前k点

        call now(time_start)
        ! 调用封装的子程序计算单个k点的Berry曲率和轨道磁化强度
        call Berry_curvature_orb_mag_singlek_allbands_pack(k, Omega_BerryCurv, m_OrbMag, W)
       
        ! 对Berry曲率进行两种方式的求和
        do i=1, 3  ! 遍历三个分量
           ! 方式1：对占据能带求和
           Omega_allk_Occ(1, i, ik) = sum(Omega_BerryCurv(1:NumOccupied, i))
           m_OrbMag_allk_Occ(1, i, ik) = sum(m_OrbMag(1:NumOccupied, i))
           ! 方式2：仅取最高占据能带
           Omega_allk_Occ(2, i, ik) = Omega_BerryCurv(NumOccupied, i)
           m_OrbMag_allk_Occ(2, i, ik) = m_OrbMag(NumOccupied, i)
        enddo

        !> 考虑费米分布的展宽效应
        if (Fermi_broadening<eps6) Fermi_broadening= eps6  ! 防止除零
        beta_fake= 1d0/Fermi_broadening  ! 计算假想温度倒数
        
        ! 应用费米分布权重
        do i=1, 3  ! 遍历三个分量
           do m= 1, Num_wann  ! 遍历所有能带
              ! 方式1：使用费米-狄拉克分布
              Omega_allk_EF(1, i, ik)= Omega_allk_EF(1, i, ik)+ &
                 Omega_BerryCurv(m, i)*fermi(W(m)-iso_energy, beta_fake)
              m_OrbMag_allk_EF(1, i, ik)= m_OrbMag_allk_EF(1, i, ik)+ &
                 m_OrbMag(m, i)*fermi(W(m)-iso_energy, beta_fake)
              ! 方式2：使用δ函数近似
              Omega_allk_EF(2, i, ik)= Omega_allk_EF(2, i, ik)+ &
                 Omega_BerryCurv(m, i)*delta(Fermi_broadening, W(m)-iso_energy)
              m_OrbMag_allk_EF(2, i, ik)= m_OrbMag_allk_EF(2, i, ik)+ &
                 m_OrbMag(m, i)*delta(Fermi_broadening, W(m)-iso_energy)
           enddo
        enddo

        call now(time_end)  ! 记录结束时间
     enddo ! ik循环

#if defined (MPI)
     ! MPI归约所有结果到主进程
     call mpi_allreduce(Omega_allk_Occ,Omega_allk_Occ_mpi,size(Omega_allk_Occ_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
     call mpi_allreduce(m_OrbMag_allk_Occ,m_OrbMag_allk_Occ_mpi,size(m_OrbMag_allk_Occ_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
     call mpi_allreduce(Omega_allk_EF,Omega_allk_EF_mpi,size(Omega_allk_EF_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
     call mpi_allreduce(m_OrbMag_allk_EF,m_OrbMag_allk_EF_mpi,size(m_OrbMag_allk_EF_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
#else
     ! 串行版本：直接复制
     Omega_allk_Occ_mpi= Omega_allk_Occ
     m_OrbMag_allk_Occ_mpi= m_OrbMag_allk_Occ
     Omega_allk_EF_mpi= Omega_allk_EF
     m_OrbMag_allk_EF_mpi= m_OrbMag_allk_EF
#endif

     !> 将Berry曲率写入文件
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature.dat')
        ! 写入文件头信息
        write(outfileindex, '(a)')'# Berry curvature in unit of (Angstrom^2)  '
        write(outfileindex, '("#", 30X,4a36)')'Sum over 1-NumOccupied bands', &
           "values at NumOccupied'th band", &  ! 最高占据能带的值
           ' Sum below Fermi level','values at Fermi level'  ! 费米能级以下求和及在费米能级处的值
        ! 写入列标题
        write(outfileindex, '(a10,2000a12)')'# kx (1/A)', 'ky (1/A)', 'kz (1/A)', &
                                            "kx' (1/A)", "ky' (1/A)", "kz' (1/A)", &  ! 旋转后的k坐标
            'Omega_x', 'Omega_y', 'Omega_z', &  ! 占据能带求和的Berry曲率
            'Omega_x', 'Omega_y', 'Omega_z', &  ! 最高占据能带的Berry曲率
            'Omega_x', 'Omega_y', 'Omega_z', &  ! 费米能级以下求和的Berry曲率
            'Omega_x', 'Omega_y', 'Omega_z'     ! 费米能级处的Berry曲率
        write(outfileindex, '("#col ", i5, 200i12)')(i, i=1,18)  ! 列号

        ! 写入数据
        ik= 0
        do i= 1, nk1
           do j= 1, nk2
              ik= ik+ 1
              ! 将k点旋转到平面内坐标系
              call rotate_k3_to_kplane(kslice_xyz(:, ik), kxy_plane)
              ! 写入完整的数据行
              write(outfileindex, '(6f12.6,2000E12.4)')kslice_xyz(:, ik)*Angstrom2atomic, &  ! 原始k坐标
                 kxy_plane*Angstrom2atomic, &  ! 旋转后的k坐标
                 ! 四种不同方式的Berry曲率（转换单位）
                 Omega_allk_Occ_mpi(1, :, ik)/Angstrom2atomic**2, &  ! 占据能带求和
                 Omega_allk_Occ_mpi(2, :, ik)/Angstrom2atomic**2, &  ! 最高占据能带
                 Omega_allk_EF_mpi(1, :, ik)/Angstrom2atomic**2, &    ! 费米能级以下求和
                 Omega_allk_EF_mpi(2, :, ik)/Angstrom2atomic**2       ! 费米能级处
           enddo
           write(outfileindex, *) ' '  ! 每行后空行
        enddo

        close(outfileindex)
     endif

     !> 转换为玻尔磁子单位
     ! 转换公式：乘以 eV2Hartree * Ang2Bohr^2 * 2
     m_OrbMag_allk_EF_mpi= m_OrbMag_allk_EF_mpi*eV2Hartree*Ang2Bohr**2 * 2
     m_OrbMag_allk_Occ_mpi= m_OrbMag_allk_Occ_mpi*eV2Hartree*Ang2Bohr**2 * 2

     ! [后续代码继续... 包括轨道磁化强度输出、各种绘图脚本生成等]

     #if defined (MPI)
     ! MPI进程同步屏障，确保所有进程完成后再继续执行
     call mpi_barrier(mpi_cmw, ierr)
#endif

     ! 释放动态分配的内存
     deallocate( kslice)        ! 释放k点切片数组
     deallocate( kslice_xyz)    ! 释放k点笛卡尔坐标数组

     return
  end subroutine Berry_curvature_plane_full  ! 结束平面Berry曲率计算子程序

  ! 子程序：封装的单k点Berry曲率和轨道磁化强度计算（所有能带）
  subroutine Berry_curvature_orb_mag_singlek_allbands_pack(k, Omega_BerryCurv, m_OrbMag, W)
     !> 为选定能带计算Berry曲率和轨道磁化强度
     !> 参考文献：Physical Review B 74, 195118(2006) 公式(34)
     !> 作者：Quansheng Wu @ EPFL, 2020年9月29日
     ! 版权所有 (c) 2020 QuanSheng Wu

     use para, only : Num_wann, dp, zi, NumOccupied  ! 只引入需要的模块变量
     implicit none

     !> 输入：分数坐标下的k点坐标
     real(dp), intent(in) :: k(3)
     
     !> 输出：Berry曲率和轨道磁化强度 (3个分量, 能带)
     real(dp), intent(out) :: m_OrbMag(Num_wann, 3)
     real(dp), intent(out) :: Omega_BerryCurv(Num_wann, 3)

     !> 输出：本征值
     real(dp), intent(out) :: W(Num_wann)
    
     integer :: i  ! 循环计数器

     !> Wannier基下的速度矩阵 (能带×能带×3方向)
     complex(dp), allocatable :: Vmn_wann(:, :, :)

     !> 哈密顿量基下的速度矩阵
     complex(dp), allocatable :: Vmn_Ham(:, :, :), Vmn_Ham_nondiag(:, :, :)
     complex(dp), allocatable :: Dmn_Ham(:, :, :)  ! Berry连接矩阵

     !> 哈密顿量、本征值和本征向量
     complex(dp), allocatable :: UU(:, :)
    
     ! 分配内存空间
     allocate(Vmn_wann(Num_wann, Num_wann, 3), Vmn_Ham(Num_wann, Num_wann, 3))
     allocate(Dmn_Ham(Num_wann,Num_wann,3), Vmn_Ham_nondiag(Num_wann, Num_wann, 3))
     allocate(UU(Num_wann, Num_wann))
     
     ! 初始化输出数组
     W=0d0
     m_OrbMag=0d0
     Omega_BerryCurv= 0d0

     !> 通过HmnR的直接傅里叶变换计算体态哈密顿量
     call ham_bulk_atomicgauge(k, UU)

     !> 调用LAPACK的zheev进行对角化
     call eigensystem_c( 'V', 'U', Num_wann, UU, W)

     !> 在Wannier基下获取速度算符
     call dHdk_atomicgauge(k, Vmn_wann)
     
     !> 将Vmn_wann从Wannier基旋转到哈密顿量基
     do i=1, 3  ! 遍历三个笛卡尔方向
        call rotation_to_Ham_basis(UU, Vmn_wann(:, :, i), Vmn_Ham(:, :, i))
     enddo

     ! 计算Berry连接矩阵和非对角速度矩阵元
     call get_Dmn_Ham(W, Vmn_Ham, Dmn_Ham)
     call get_Vmn_Ham_nondiag(Vmn_Ham, Vmn_Ham_nondiag)

     ! 计算Berry曲率和轨道磁化强度
     call Berry_curvature_singlek_allbands(Dmn_Ham, Omega_BerryCurv)
     call orbital_magnetization_singlek_allbands(Dmn_Ham, Vmn_Ham_nondiag, m_OrbMag)

     ! 释放内存
     deallocate(Vmn_wann, Vmn_Ham, Vmn_Ham_nondiag)
     deallocate(UU, Dmn_Ham)
     return
  end subroutine Berry_curvature_orb_mag_singlek_allbands_pack

  ! 子程序：基于费米能级计算平面Berry曲率和轨道磁化强度
  subroutine Berry_curvature_plane_EF
     !> 为选定能带计算基于费米能级的Berry曲率和轨道磁化强度
     !> 参考文献：Physical Review B 74, 195118(2006) 公式(34)
     !> 作者：Quansheng Wu @ EPFL, 2020年8月20日
     ! 版权所有 (c) 2020 QuanSheng Wu

     use wmpi
     use para
     implicit none
    
     ! 局部变量
     integer :: ik, i, j, n, ierr, nkmesh2
     real(dp) :: k(3), o1(3), vmin, vmax  ! k点和辅助变量
     !> k点切片
     real(dp), allocatable :: kslice(:, :), kslice_xyz(:, :)
     real(dp), external :: norm  ! 外部范数函数
     real(dp) :: time_start, time_end, time_start0  ! 时间统计

     !> Berry曲率和轨道磁化强度数组 (能带, 3分量, k点)
     real(dp), allocatable :: Omega_allk(:, :, :), Omega_allk_mpi(:, :, :)
     real(dp), allocatable :: m_OrbMag_allk(:, :, :), m_OrbMag_allk_mpi(:, :, :)
     real(dp), allocatable :: Omega_BerryCurv(:, :), m_OrbMag(:, :)

     !> Wannier基下的速度矩阵
     complex(dp), allocatable :: Vmn_wann(:, :, :)

     !> 哈密顿量基下的速度矩阵
     complex(dp), allocatable :: Vmn_Ham(:, :, :), Vmn_Ham_nondiag(:, :, :)
     complex(dp), allocatable :: Dmn_Ham(:, :, :)

     !> 哈密顿量、本征值和本征向量
     complex(dp), allocatable :: UU(:, :)
     real(dp), allocatable :: W(:)  ! 本征值数组

     ! 初始化网格大小和数组分配
     nkmesh2= Nk1*Nk2  ! 二维k点网格总数
     ! 分配各种数组内存
     allocate(Vmn_wann(Num_wann, Num_wann, 3), Vmn_Ham(Num_wann, Num_wann, 3))
     allocate(Dmn_Ham(Num_wann,Num_wann,3), Vmn_Ham_nondiag(Num_wann, Num_wann, 3))
     allocate(W(Num_wann))
     allocate(UU(Num_wann, Num_wann))
     allocate( Omega_BerryCurv(Num_wann, 3), m_OrbMag(Num_wann, 3))
     allocate( Omega_allk    (Num_wann, 3, nkmesh2))
     allocate( Omega_allk_mpi(Num_wann, 3, nkmesh2))
     allocate( m_OrbMag_allk    (Num_wann, 3, nkmesh2))
     allocate( m_OrbMag_allk_mpi(Num_wann, 3, nkmesh2))
     allocate( kslice(3, nkmesh2))
     allocate( kslice_xyz(3, nkmesh2))
     
     ! 初始化数组为零
     m_OrbMag=0d0
     Omega_allk= 0d0
     m_OrbMag_allk=0d0
     Omega_allk_mpi= 0d0
     Omega_BerryCurv= 0d0
     m_OrbMag_allk_mpi=0d0
     kslice=0d0
     kslice_xyz=0d0
    
     !> 构建以K3d_start为中心的k点切片
     ik =0
     do i= 1, nk1  ! k1方向循环
        do j= 1, nk2  ! k2方向循环
           ik =ik +1
           ! 构建分数坐标k点网格（居中于K3D_start）
           kslice(:, ik)= K3D_start+ K3D_vec1*(i-1)/dble(nk1-1)  &
                     + K3D_vec2*(j-1)/dble(nk2-1) - (K3D_vec1+ K3D_vec2)/2d0
           ! 转换为笛卡尔坐标
           kslice_xyz(:, ik)= kslice(1, ik)* Origin_cell%Kua+ kslice(2, ik)* Origin_cell%Kub+ kslice(3, ik)* Origin_cell%Kuc 
        enddo
     enddo

     ! 时间统计初始化
     time_start= 0d0
     time_start0= 0d0
     call now(time_start0)  ! 获取开始时间
     time_start= time_start0
     time_end  = time_start0
     
     ! 主循环：遍历二维k点网格（并行）
     do ik= 1+ cpuid, nkmesh2, num_cpu
        ! 进度输出
        if (cpuid==0.and. mod(ik/num_cpu, 100)==0) &
           write(stdout, '(a, i9, "  /", i10, a, f10.1, "s", a, f10.1, "s")') &
           ' Berry curvature: ik', ik, nkmesh2, ' time left', &
           (nkmesh2-ik)*(time_end- time_start)/num_cpu, &  ! 预估剩余时间
           ' time elapsed: ', time_end-time_start0            ! 已用时间

        call now(time_start)  ! 记录开始时间
 
        !> 对角化哈密顿量
        k= kslice(:, ik)  ! 当前k点

        call now(time_start)
        
        !> 通过HmnR的直接傅里叶变换计算体态哈密顿量
        call ham_bulk_atomicgauge(k, UU)
   
        !> 调用LAPACK进行对角化
        call eigensystem_c( 'V', 'U', Num_wann, UU, W)

        !> 在Wannier基下获取速度算符
        call dHdk_atomicgauge(k, Vmn_wann)
        
        !> 将Vmn_wann从Wannier基旋转到哈密顿量基
        do i=1, 3  ! 遍历三个笛卡尔方向
           call rotation_to_Ham_basis(UU, Vmn_wann(:, :, i), Vmn_Ham(:, :, i))
        enddo

        ! 计算Berry连接矩阵和非对角速度矩阵元
        call get_Dmn_Ham(W, Vmn_Ham, Dmn_Ham)
        call get_Vmn_Ham_nondiag(Vmn_Ham, Vmn_Ham_nondiag)

        ! 计算单个k点的Berry曲率和轨道磁化强度
        call Berry_curvature_singlek_allbands(Dmn_Ham, Omega_BerryCurv)
        call orbital_magnetization_singlek_allbands(Dmn_Ham, Vmn_Ham_nondiag, m_OrbMag)
        
        ! 保存结果
        Omega_allk(:, :, ik) = Omega_BerryCurv
        m_OrbMag_allk(:, :, ik) = m_OrbMag

        call now(time_end)  ! 记录结束时间
     enddo ! ik循环

#if defined (MPI)
     ! MPI归约所有结果到主进程
     call mpi_allreduce(Omega_allk,Omega_allk_mpi,size(Omega_allk_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
     call mpi_allreduce(m_OrbMag_allk,m_OrbMag_allk_mpi,size(m_OrbMag_allk_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
#else
     ! 串行版本：直接复制
     Omega_allk_mpi= Omega_allk
     m_OrbMag_allk_mpi= m_OrbMag_allk
#endif

     !> 输出Berry曲率和轨道磁化强度到文件
     outfileindex= outfileindex+ 1
     if (cpuid==0) then  ! 主进程输出
        open(unit=outfileindex, file='Berrycurvature_Orbitalmagnetization.dat')
        ! 写入列号信息
        write(outfileindex, '("#col ", i5, 200i12)')(i, i=1, NumberofSelectedBands*6+3)
        ! 写入列标题
        write(outfileindex, '(a10,2000a12)')'# kx (1/A)', 'ky (1/A)', 'kz (1/A)', &
           'Omega_x(A^2)', 'Omega_y(A^2)', 'Omega_z(A^2)' , 'm_x', 'm_y', 'm_z'

        ! 写入数据
        ik= 0
        do i= 1, nk1
           do j= 1, nk2
              ik= ik+ 1
              ! 写入k点坐标和选定能带的Berry曲率与轨道磁化强度
              write(outfileindex, '(3f12.6,2000E12.4)')kslice_xyz(:, ik)*Angstrom2atomic, &
                 ! 遍历选定的能带索引
                 (Omega_allk_mpi(Selected_band_index(n), :, ik)/Angstrom2atomic**2, &  ! Berry曲率（转换单位）
                 m_OrbMag_allk_mpi(Selected_band_index(n), :, ik), n=1, NumberofSelectedBands)   
           enddo
           write(outfileindex, *) ' '  ! 每行后空行分隔
        enddo

        close(outfileindex)  ! 关闭文件
     endif

     !> 生成gnuplot脚本绘制Berry曲率图
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature.gnu')
        ! 设置gnuplot参数
        write(outfileindex, '(a)')"set encoding iso_8859_1"
        ! 设置终端为pngcairo（高质量PNG）
        write(outfileindex, '(a)')'set terminal  pngcairo  truecolor enhanced size 3680, 1920 font ",40"'
        write(outfileindex, '(a)')'#set terminal  png       truecolor enhanced size 3680, 1920 font ",40"'
        write(outfileindex, '(a)')"set output 'Berrycurvature.png'"
        ! 设置多图布局参数
        write(outfileindex, '(a)')'if (!exists("MP_LEFT"))   MP_LEFT = .12'
        write(outfileindex, '(a)')'if (!exists("MP_RIGHT"))  MP_RIGHT = .92'
        write(outfileindex, '(a)')'if (!exists("MP_BOTTOM")) MP_BOTTOM = .12'
        write(outfileindex, '(a)')'if (!exists("MP_TOP"))    MP_TOP = .88'
        write(outfileindex, '(a)')'if (!exists("MP_GAP"))    MP_GAP = 0.08'
        write(outfileindex, '(a)')'set multiplot layout 1,3 rowsfirst \'
        write(outfileindex, '(a)')"              margins screen MP_LEFT, MP_RIGHT, MP_BOTTOM, MP_TOP spacing screen MP_GAP"
        write(outfileindex, '(a)')" "
        ! 设置绘图样式
        write(outfileindex, '(a)')"set palette rgbformulae 33,13,10"
        write(outfileindex, '(a)')"unset ztics"
        write(outfileindex, '(a)')"unset key"
        write(outfileindex, '(a)')"set pm3d"
        write(outfileindex, '(a)')"#set zbrange [ -10: 10] "
        write(outfileindex, '(a, f10.3, a, f10.3, a)')"#set cbrange [ ", vmin, ':', vmax, " ] "
        write(outfileindex, '(a)')"set view map"
        write(outfileindex, '(a)')"set size ratio -1"
        write(outfileindex, '(a)')"set border lw 3"
        write(outfileindex, '(a)')"set xlabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"set ylabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"unset colorbox"
        write(outfileindex, '(a)')"#unset xtics"
        write(outfileindex, '(a)')"#unset xlabel"
        write(outfileindex, '(a)')"set xrange [] noextend"
        write(outfileindex, '(a)')"set yrange [] noextend"
        write(outfileindex, '(a)')"set ytics 0.5 nomirror scale 0.5"
        write(outfileindex, '(a)')"set pm3d interpolate 2,2"
        ! 绘制三个方向的Berry曲率分量
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_x'"
        write(outfileindex, '(a)')"splot 'Berrycurvature_Orbitalmagnetization.dat' u 1:2:4 w pm3d"
        write(outfileindex, '(a)')"unset ylabel"
        write(outfileindex, '(a)')"unset ytics"
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_y'"
        write(outfileindex, '(a)')"splot 'Berrycurvature_Orbitalmagnetization.dat' u 1:2:5 w pm3d"
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_z'"
        write(outfileindex, '(a)')"set colorbox"
        write(outfileindex, '(a)')"splot 'Berrycurvature_Orbitalmagnetization.dat' u 1:2:6 w pm3d"
        close(outfileindex)  ! 关闭gnuplot脚本文件
     endif

#if defined (MPI)
     ! MPI进程同步
     call mpi_barrier(mpi_cmw, ierr)
#endif

     ! 释放内存
     deallocate( kslice)
     deallocate( kslice_xyz)
 
     return
  end subroutine Berry_curvature_plane_EF

  ! 子程序：计算选定能带的平面Berry曲率和轨道磁化强度
  subroutine Berry_curvature_plane_selectedbands
     !> 为选定能带计算Berry曲率和轨道磁化强度
     !> 参考文献：Physical Review B 74, 195118(2006) 公式(34)
     !> 作者：Quansheng Wu @ EPFL, 2020年8月20日
     ! 版权所有 (c) 2020 QuanSheng Wu

     use wmpi
     use para
     implicit none
    
     ! 局部变量（与上一个子程序类似）
     integer :: ik, i, j, n, ierr, nkmesh2
     real(dp) :: k(3), o1(3), vmin, vmax
     !> k点切片
     real(dp), allocatable :: kslice(:, :), kslice_xyz(:, :)
     real(dp), external :: norm
     real(dp) :: time_start, time_end, time_start0

     !> Berry曲率和轨道磁化强度数组
     real(dp), allocatable :: Omega_allk(:, :, :), Omega_allk_mpi(:, :, :)
     real(dp), allocatable :: m_OrbMag_allk(:, :, :), m_OrbMag_allk_mpi(:, :, :)
     real(dp), allocatable :: Omega_BerryCurv(:, :), m_OrbMag(:, :)

     !> 速度矩阵相关数组
     complex(dp), allocatable :: Vmn_wann(:, :, :)
     complex(dp), allocatable :: Vmn_Ham(:, :, :), Vmn_Ham_nondiag(:, :, :)
     complex(dp), allocatable :: Dmn_Ham(:, :, :)

     !> 哈密顿量和本征值
     complex(dp), allocatable :: UU(:, :)
     real(dp), allocatable :: W(:)

     ! 内存分配和初始化（与上一个子程序完全相同）
     nkmesh2= Nk1*Nk2
     allocate(Vmn_wann(Num_wann, Num_wann, 3), Vmn_Ham(Num_wann, Num_wann, 3))
     allocate(Dmn_Ham(Num_wann,Num_wann,3), Vmn_Ham_nondiag(Num_wann, Num_wann, 3))
     allocate(W(Num_wann))
     allocate(UU(Num_wann, Num_wann))
     allocate( Omega_BerryCurv(Num_wann, 3), m_OrbMag(Num_wann, 3))
     allocate( Omega_allk    (Num_wann, 3, nkmesh2))
     allocate( Omega_allk_mpi(Num_wann, 3, nkmesh2))
     allocate( m_OrbMag_allk    (Num_wann, 3, nkmesh2))
     allocate( m_OrbMag_allk_mpi(Num_wann, 3, nkmesh2))
     allocate( kslice(3, nkmesh2))
     allocate( kslice_xyz(3, nkmesh2))
     m_OrbMag=0d0
     Omega_allk= 0d0
     m_OrbMag_allk=0d0
     Omega_allk_mpi= 0d0
     Omega_BerryCurv= 0d0
     m_OrbMag_allk_mpi=0d0

     kslice=0d0
     kslice_xyz=0d0
    
     !> 构建k点切片（与上一个子程序相同）
     ik =0
     do i= 1, nk1
        do j= 1, nk2
           ik =ik +1
           kslice(:, ik)= K3D_start+ K3D_vec1*(i-1)/dble(nk1-1)  &
                     + K3D_vec2*(j-1)/dble(nk2-1) - (K3D_vec1+ K3D_vec2)/2d0
           kslice_xyz(:, ik)= kslice(1, ik)* Origin_cell%Kua+ kslice(2, ik)* Origin_cell%Kub+ kslice(3, ik)* Origin_cell%Kuc 
        enddo
     enddo

     ! 时间统计和主循环（与上一个子程序完全相同）
     time_start= 0d0
     time_start0= 0d0
     call now(time_start0)
     time_start= time_start0
     time_end  = time_start0
     do ik= 1+ cpuid, nkmesh2, num_cpu
        if (cpuid==0.and. mod(ik/num_cpu, 100)==0) &
           write(stdout, '(a, i9, "  /", i10, a, f10.1, "s", a, f10.1, "s")') &
           ' Berry curvature: ik', ik, nkmesh2, ' time left', &
           (nkmesh2-ik)*(time_end- time_start)/num_cpu, &
           ' time elapsed: ', time_end-time_start0 

        call now(time_start)
 
        !> 对角化哈密顿量
        k= kslice(:, ik)

        call now(time_start)
        
        !> 计算体态哈密顿量
        call ham_bulk_atomicgauge(k, UU)
   
        !> 对角化
        call eigensystem_c( 'V', 'U', Num_wann, UU, W)

        !> 获取Wannier基速度算符
        call dHdk_atomicgauge(k, Vmn_wann)
        
        !> 旋转到哈密顿量基
        do i=1, 3
           call rotation_to_Ham_basis(UU, Vmn_wann(:, :, i), Vmn_Ham(:, :, i))
        enddo

        call get_Dmn_Ham(W, Vmn_Ham, Dmn_Ham)
        call get_Vmn_Ham_nondiag(Vmn_Ham, Vmn_Ham_nondiag)

        call Berry_curvature_singlek_allbands(Dmn_Ham, Omega_BerryCurv)
        call orbital_magnetization_singlek_allbands(Dmn_Ham, Vmn_Ham_nondiag, m_OrbMag)
        Omega_allk(:, :, ik) = Omega_BerryCurv
        m_OrbMag_allk(:, :, ik) = m_OrbMag

        call now(time_end)
     enddo ! ik

#if defined (MPI)
     ! MPI归约（与上一个子程序相同）
     call mpi_allreduce(Omega_allk,Omega_allk_mpi,size(Omega_allk_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
     call mpi_allreduce(m_OrbMag_allk,m_OrbMag_allk_mpi,size(m_OrbMag_allk_mpi),&
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
#else
     Omega_allk_mpi= Omega_allk
     m_OrbMag_allk_mpi= m_OrbMag_allk
#endif

     !> 输出选定能带的Berry曲率和轨道磁化强度
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature_Orbitalmagnetization.dat')
        write(outfileindex, '("#col ", i5, 200i12)')(i, i=1, NumberofSelectedBands*6+3)
        write(outfileindex, '(a10,2000a12)')'# kx (1/A)', 'ky (1/A)', 'kz (1/A)', &
           'Omega_x', 'Omega_y', 'Omega_z' , 'm_x', 'm_y', 'm_z'

        ik= 0
        do i= 1, nk1
           do j= 1, nk2
              ik= ik+ 1
              ! 写入选定能带的数据（注意：这里没有单位转换）
              write(outfileindex, '(3f12.6,2000E12.4)')kslice_xyz(:, ik)*Angstrom2atomic, &
                 (Omega_allk_mpi(Selected_band_index(n), :, ik)/Angstrom2atomic**2, &
                 m_OrbMag_allk_mpi(Selected_band_index(n), :, ik), n=1, NumberofSelectedBands)   
           enddo
           write(outfileindex, *) ' '
        enddo

        close(outfileindex)
     endif

     !> 生成gnuplot脚本（与上一个子程序相同）
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        ! [gnuplot脚本内容与上一个子程序完全相同，此处省略重复注释]
        open(unit=outfileindex, file='Berrycurvature.gnu')
        write(outfileindex, '(a)')"set encoding iso_8859_1"
        write(outfileindex, '(a)')'set terminal  pngcairo  truecolor enhanced size 3680, 1920 font ",40"'
        write(outfileindex, '(a)')'#set terminal  png       truecolor enhanced size 3680, 1920 font ",40"'
        write(outfileindex, '(a)')"set output 'Berrycurvature.png'"
        write(outfileindex, '(a)')'if (!exists("MP_LEFT"))   MP_LEFT = .12'
        write(outfileindex, '(a)')'if (!exists("MP_RIGHT"))  MP_RIGHT = .92'
        write(outfileindex, '(a)')'if (!exists("MP_BOTTOM")) MP_BOTTOM = .12'
        write(outfileindex, '(a)')'if (!exists("MP_TOP"))    MP_TOP = .88'
        write(outfileindex, '(a)')'if (!exists("MP_GAP"))    MP_GAP = 0.08'
        write(outfileindex, '(a)')'set multiplot layout 1,3 rowsfirst \'
        write(outfileindex, '(a)')"              margins screen MP_LEFT, MP_RIGHT, MP_BOTTOM, MP_TOP spacing screen MP_GAP"
        write(outfileindex, '(a)')" "
        write(outfileindex, '(a)')"set palette rgbformulae 33,13,10"
        write(outfileindex, '(a)')"unset ztics"
        write(outfileindex, '(a)')"unset key"
        write(outfileindex, '(a)')"set pm3d"
        write(outfileindex, '(a)')"#set zbrange [ -10: 10] "
        write(outfileindex, '(a, f10.3, a, f10.3, a)')"#set cbrange [ ", vmin, ':', vmax, " ] "
        write(outfileindex, '(a)')"set view map"
        write(outfileindex, '(a)')"set size ratio -1"
        write(outfileindex, '(a)')"set border lw 3"
        write(outfileindex, '(a)')"set xlabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"set ylabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"unset colorbox"
        write(outfileindex, '(a)')"#unset xtics"
        write(outfileindex, '(a)')"#unset xlabel"
        write(outfileindex, '(a)')"set xrange [] noextend"
        write(outfileindex, '(a)')"set yrange [] noextend"
        write(outfileindex, '(a)')"set ytics 0.5 nomirror scale 0.5"
        write(outfileindex, '(a)')"set pm3d interpolate 2,2"
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_x'"
        write(outfileindex, '(a)')"splot 'Berrycurvature_Orbitalmagnetization.dat' u 1:2:4 w pm3d"
        write(outfileindex, '(a)')"unset ylabel"
        write(outfileindex, '(a)')"unset ytics"
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_y'"
        write(outfileindex, '(a)')"splot 'Berrycurvature_Orbitalmagnetization.dat' u 1:2:5 w pm3d"
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_z'"
        write(outfileindex, '(a)')"set colorbox"
        write(outfileindex, '(a)')"splot 'Berrycurvature_Orbitalmagnetization.dat' u 1:2:6 w pm3d"
        close(outfileindex)
     endif
     #if defined (MPI)
     ! MPI进程同步屏障，确保所有进程完成后再继续执行
     call mpi_barrier(mpi_cmw, ierr)
#endif

     ! 释放动态分配的内存
     deallocate( kslice)        ! 释放k点切片数组
     deallocate( kslice_xyz)    ! 释放k点笛卡尔坐标数组
 
     return
  end subroutine Berry_curvature_plane_selectedbands  ! 结束选定能带Berry曲率计算子程序

  ! 子程序：计算平面Berry曲率（原始版本）
  subroutine Berry_curvature_plane
     !> 计算Berry曲率
     !> 参考文献：Physical Review B 74, 195118(2006) 公式(34)
     !> 作者：Quansheng Wu @ ETHZ, 2015年9月22日
     ! 版权所有 (c) 2010 QuanSheng Wu

     use wmpi
     use para
     implicit none
    
     ! 局部变量
     integer :: ik, i, j, ierr
     real(dp) :: k(3), o1(3), vmin, vmax  ! k点和辅助变量

     !> k点切片
     real(dp), allocatable :: kslice(:, :), kslice_xyz(:, :)
     real(dp), external :: norm  ! 外部范数函数
     real(dp) :: time_start, time_end, time_start0  ! 时间统计

     !> Berry曲率数组 (3方向, k点)
     complex(dp), allocatable :: Omega_x(:), Omega_y(:), Omega_z(:)  ! 各能带的Berry曲率
     complex(dp), allocatable :: Omega(:, :), Omega_mpi(:, :)        ! 总Berry曲率和MPI归约结果

     ! 分配内存
     allocate( kslice(3, Nk1*Nk2))
     allocate( kslice_xyz(3, Nk1*Nk2))
     allocate( Omega_x(Num_wann))    ! x方向Berry曲率（每个能带）
     allocate( Omega_y(Num_wann))    ! y方向Berry曲率（每个能带）
     allocate( Omega_z(Num_wann))    ! z方向Berry曲率（每个能带）
     allocate( Omega    (3, Nk1*Nk2)) ! 总Berry曲率（3方向×k点）
     allocate( Omega_mpi(3, Nk1*Nk2)) ! MPI归约后的Berry曲率
     
     ! 初始化数组
     kslice=0d0
     kslice_xyz=0d0
     omega= 0d0
     omega_mpi= 0d0
    
     !> 构建以K3d_start为中心的k点切片
     ik =0
     do i= 1, nk1  ! k1方向循环
        do j= 1, nk2  ! k2方向循环
           ik =ik +1
           ! 构建分数坐标k点网格（居中于K3D_start）
           kslice(:, ik)= K3D_start+ K3D_vec1*(i-1)/dble(nk1-1)  &
                     + K3D_vec2*(j-1)/dble(nk2-1) - (K3D_vec1+ K3D_vec2)/2d0
           ! 转换为笛卡尔坐标
           kslice_xyz(:, ik)= kslice(1, ik)* Origin_cell%Kua+ kslice(2, ik)* Origin_cell%Kub+ kslice(3, ik)* Origin_cell%Kuc 
        enddo
     enddo

     ! 时间统计初始化
     time_start= 0d0
     time_start0= 0d0
     call now(time_start0)  ! 获取开始时间
     time_start= time_start0
     time_end  = time_start0
     
     ! 主循环：遍历二维k点网格（并行）
     do ik= 1+ cpuid, Nk1*Nk2, num_cpu
        ! 进度输出
        if (cpuid==0.and. mod((ik-1)/num_cpu, 100)==0) &
           write(stdout, '(a, i9, "  /", i10, a, f10.1, "s", a, f10.1, "s")') &
           ' Berry curvature: ik', ik, Nk1*Nk2, ' time left', &
           (nk1*nk2-ik)*(time_end- time_start)/num_cpu, &  ! 预估剩余时间
           ' time elapsed: ', time_end-time_start0            ! 已用时间

        call now(time_start)  ! 记录开始时间
 
        !> 对角化哈密顿量
        k= kslice(:, ik)  ! 当前k点

        ! 初始化Berry曲率数组
        Omega_x= 0d0
        Omega_y= 0d0
        Omega_z= 0d0

        ! 根据标志选择不同的Berry曲率计算方法
        if (Berrycurvature_EF_calc) then
           ! 基于费米能级的计算
           call Berry_curvature_singlek_EF(k, iso_energy, Omega_x, Omega_y, Omega_z)
        else
           ! 基于占据数的计算（只计算第一个能带）
           call Berry_curvature_singlek_numoccupied_total(k, Omega_x(1), Omega_y(1), Omega_z(1))
        endif
        
        ! 对能带求和得到总Berry曲率
        Omega(1, ik) = sum(Omega_x)  ! x方向总和
        Omega(2, ik) = sum(Omega_y)  ! y方向总和  
        Omega(3, ik) = sum(Omega_z)  ! z方向总和
        
        call now(time_end)  ! 记录结束时间
     enddo ! ik循环

     ! 初始化MPI归约数组
     Omega_mpi= 0d0

#if defined (MPI)
     ! MPI归约所有进程的Berry曲率结果
     call mpi_allreduce(Omega,Omega_mpi,size(Omega_mpi),&
                       mpi_dc,mpi_sum,mpi_cmw,ierr)
#else
     ! 串行版本：直接复制
     Omega_mpi= Omega
#endif
     
     ! 计算Berry曲率的实部的统计范围
     vmax=maxval(real(Omega_mpi))  ! 最大值
     vmin=minval(real(Omega_mpi))  ! 最小值

     !> 输出原始Berry曲率数据到文件
     outfileindex= outfileindex+ 1
     if (cpuid==0) then  ! 主进程输出
        open(unit=outfileindex, file='Berrycurvature.dat')
        ! 写入说明信息
        write(outfileindex, '(20a28)')'# Please take the real part for your use'
        write(outfileindex, '(20a28)')'# kx (1/A)', 'ky (1/A)', 'kz (1/A)', &
           'Omega_x', 'Omega_y', 'Omega_z' 

        ! 写入数据
        ik= 0
        do i= 1, nk1
           do j= 1, nk2
              ik= ik+ 1
              ! 写入k点坐标和Berry曲率实部（转换单位）
              write(outfileindex, '(20E28.10)')kslice_xyz(:, ik)*Angstrom2atomic, &
                 real(Omega_mpi(:, ik))/Angstrom2atomic**2
           enddo
           write(outfileindex, *) ' '  ! 每行后空行分隔
        enddo

        close(outfileindex)  ! 关闭文件
     endif

     !> 输出归一化的Berry曲率数据到文件
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature-normalized.dat')
        write(outfileindex, '(20a28)')'# Please take the real part for your use'
        write(outfileindex, '(20a28)')'# kx (1/A)', 'ky (1/A)', 'kz (1/A)', &
           'Omega_x(A^2)', 'Omega_y(A^2)', 'Omega_z(A^2)'

        ik= 0
        do i= 1, nk1
           do j= 1, nk2
              ik= ik+ 1
              o1= real(Omega_mpi(:,ik))  ! 获取实部的Berry曲率矢量
              ! 归一化处理（避免除零）
              if (norm(o1)>eps9) o1= o1/norm(o1)
              write(outfileindex, '(20f28.10)')kslice_xyz(:, ik)*Angstrom2atomic, o1
           enddo
           write(outfileindex, *) ' '
        enddo
        close(outfileindex)
     endif

     !> 生成原始Berry曲率的gnuplot脚本
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature.gnu')
        ! 设置gnuplot参数
        write(outfileindex, '(a)')"set encoding iso_8859_1"
        write(outfileindex, '(a)')'set terminal  pngcairo  truecolor enhanced size 3680, 1920 font ",40"'
        write(outfileindex, '(a)')'#set terminal  png       truecolor enhanced size 3680, 1920 font ",40"'
        write(outfileindex, '(a)')"set output 'Berrycurvature.png'"
        ! 设置多图布局参数
        write(outfileindex, '(a)')'if (!exists("MP_LEFT"))   MP_LEFT = .12'
        write(outfileindex, '(a)')'if (!exists("MP_RIGHT"))  MP_RIGHT = .92'
        write(outfileindex, '(a)')'if (!exists("MP_BOTTOM")) MP_BOTTOM = .12'
        write(outfileindex, '(a)')'if (!exists("MP_TOP"))    MP_TOP = .88'
        write(outfileindex, '(a)')'if (!exists("MP_GAP"))    MP_GAP = 0.08'
        write(outfileindex, '(a)')'set multiplot layout 1,3 rowsfirst \'
        write(outfileindex, '(a)')"              margins screen MP_LEFT, MP_RIGHT, MP_BOTTOM, MP_TOP spacing screen MP_GAP"
        write(outfileindex, '(a)')" "
        write(outfileindex, '(a)')"set palette rgbformulae 33,13,10"
        write(outfileindex, '(a)')"unset ztics"
        write(outfileindex, '(a)')"unset key"
        write(outfileindex, '(a)')"set pm3d"
        write(outfileindex, '(a)')"#set zbrange [ -10: 10] "
        write(outfileindex, '(a, f10.3, a, f10.3, a)')"set cbrange [ ", vmin, ':', vmax, " ] "  ! 设置颜色范围
        write(outfileindex, '(a)')"set view map"
        write(outfileindex, '(a)')"set size ratio -1"
        write(outfileindex, '(a)')"set border lw 3"
        write(outfileindex, '(a)')"set xlabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"set ylabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"unset colorbox"
        write(outfileindex, '(a)')"#unset xtics"
        write(outfileindex, '(a)')"#unset xlabel"
        write(outfileindex, '(a)')"set xrange [] noextend"
        write(outfileindex, '(a)')"set yrange [] noextend"
        write(outfileindex, '(a)')"set ytics 0.5 nomirror scale 0.5"
        write(outfileindex, '(a)')"set pm3d interpolate 2,2"
        ! 绘制三个方向的Berry曲率分量
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_x'"
        write(outfileindex, '(a)')"splot 'Berrycurvature.dat' u 1:2:4 w pm3d"
        write(outfileindex, '(a)')"unset ylabel"
        write(outfileindex, '(a)')"unset ytics"
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_y'"
        write(outfileindex, '(a)')"splot 'Berrycurvature.dat' u 1:2:5 w pm3d"
        write(outfileindex, '(a)')"set title 'Berry Curvature {/Symbol W}_z'"
        write(outfileindex, '(a)')"set colorbox"
        write(outfileindex, '(a)')"splot 'Berrycurvature.dat' u 1:2:6 w pm3d"
        close(outfileindex)  ! 关闭gnuplot脚本文件
     endif

     !> 生成归一化Berry曲率的gnuplot脚本（矢量图）
     outfileindex= outfileindex+ 1
     if (cpuid==0) then
        open(unit=outfileindex, file='Berrycurvature-normalized.gnu')
        write(outfileindex, '(a)')"set encoding iso_8859_1"
        write(outfileindex, '(a)')'#set terminal  pngcairo  truecolor enhanced size 1920, 1680 font ",40"'
        write(outfileindex, '(a)')'set terminal  png       truecolor enhanced size 1920, 1680 font ",40"'
        write(outfileindex, '(a)')"set output 'Berrycurvature-normalized.png'"
        write(outfileindex, '(a)')"unset ztics"
        write(outfileindex, '(a)')"unset key"
        write(outfileindex, '(a)')"set border lw 3"
        write(outfileindex, '(a)')"set xlabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"set ylabel 'k (1/{\305})'"
        write(outfileindex, '(a)')"unset colorbox"
        write(outfileindex, '(a)')"unset xtics"
        write(outfileindex, '(a)')"unset xlabel"
        write(outfileindex, '(a)')"set xrange [] noextend"
        write(outfileindex, '(a)')"set yrange [] noextend"
        write(outfileindex, '(a)')"set ytics 0.5 nomirror scale 0.5"
        write(outfileindex, '(a)')"set pm3d interpolate 2,2"
        write(outfileindex, '(a)')"set title '({/Symbol W}_x, {/Symbol W}_y) real'"
        ! 绘制归一化的Berry曲率矢量场
        write(outfileindex, '(a)')"plot 'Berrycurvature-normalized.dat' u 1:2:($4/500):($5/500) w vec"
        close(outfileindex)
     endif

#if defined (MPI)
     ! MPI进程同步
     call mpi_barrier(mpi_cmw, ierr)
#endif

     ! 释放内存
     deallocate( kslice)
     deallocate( kslice_xyz)
     deallocate( Omega_x, Omega_y, Omega_z)
     deallocate( Omega, Omega_mpi)
 
     return
  end subroutine Berry_curvature_plane

  ! 子程序：从实空间哈密顿量傅里叶变换到k空间
  subroutine Fourier_R_to_k(k, ham)
     !> 将哈密顿量从实空间(R空间)傅里叶变换到k空间
     use para, only: irvec, HmnR, Nrpts, ndegen, pi, zi, Num_wann, dp
     implicit none

     real(dp), intent(in) :: k(3)                    ! 输入的k点坐标
     complex(dp), intent(out) :: ham(Num_wann, Num_wann)  ! 输出的k空间哈密顿量
     integer :: iR                                   ! 实空间格点索引
     real(dp) :: kdotr                               ! k·r点积

     ham= 0d0  ! 初始化哈密顿量为零
     ! 遍历所有实空间格点
     do iR= 1, Nrpts
        ! 计算k·r点积
        kdotr= k(1)*irvec(1,iR) + k(2)*irvec(2,iR) + k(3)*irvec(3,iR)
        ! 傅里叶变换：H(k) = Σ_R H(R) * exp(i2πk·R) / deg(R)
        Ham= Ham+ HmnR(:,:,iR)*Exp(2d0*pi*zi*kdotr)/ndegen(iR)
     enddo

     return
  end subroutine Fourier_R_to_k

  ! 子程序：计算矩阵的虚部迹
  subroutine Im_trace(ndim, A, tr)
     !> 计算矩阵A的虚部迹（仅取对角元素的虚部）
     use para, only : dp
     implicit none
     integer :: ndim                    ! 矩阵维度
     complex(dp), intent(out) :: tr     ! 输出的虚部迹
     complex(dp), intent(in) :: A(ndim, ndim)  ! 输入矩阵

     integer :: i                      ! 循环计数器

     tr = 0d0
     do i=1, ndim  ! 遍历对角元素
        tr= tr+ aimag(A(i, i))  ! 累加对角元素的虚部
     enddo

     return
  end subroutine Im_trace

  ! 子程序：计算矩阵的完整迹
  subroutine trace(ndim, A, tr)
     !> 计算矩阵A的完整迹（包括实部和虚部）
     use para, only : dp
     implicit none
     integer :: ndim                    ! 矩阵维度
     complex(dp), intent(out) :: tr     ! 输出的迹
     complex(dp), intent(in) :: A(ndim, ndim)  ! 输入矩阵

     integer :: i                      ! 循环计数器

     tr = 0d0
     do i=1, ndim  ! 遍历对角元素
        tr= tr+ (A(i, i))  ! 累加对角元素（完整复数）
     enddo

     return
  end subroutine trace

  ! 子程序：在球面上计算陈数
  subroutine Chern_sphere_single(k0, r0, Chern)
     !> 通过在球面上积分计算陈数
     !> 球面由中心k0和半径r0定义
     !> C= ∫dθdφ (...)
     !> k0必须在笛卡尔坐标系中
     !> 作者：QuanSheng Wu @ EPFL, 2018年6月12日

     use wmpi
     use para
     implicit none
     
     !> 输入输出变量
     real(dp), intent(in) :: k0(3)    ! 球面中心（笛卡尔坐标）
     real(dp), intent(in) :: r0       ! 球面半径
     real(dp), intent(out) :: Chern   ! 输出的陈数

     ! 局部变量
     integer :: ik, ik1, ik2, nkmesh2, ierr
     real(dp) :: theta, phi, r_para, dtheta, dphi, Chern_mpi  ! 球坐标和积分参数
     real(dp) :: st, ct, sp, cp, O_x, O_y, O_z  ! 三角函数和Berry曲率分量

     !> 球面上的k点
     real(dp) :: k_cart(3), k_direct(3)  ! 笛卡尔和分数坐标
     real(dp), allocatable :: kpoints(:, :)  ! 所有k点
     real(dp), allocatable :: thetas(:)      ! θ角数组
     real(dp), allocatable :: phis(:)       ! φ角数组

     !> Berry曲率数组
     complex(dp), allocatable :: Omega_x(:)
     complex(dp), allocatable :: Omega_y(:)
     complex(dp), allocatable :: Omega_z(:)

     ! 分配内存
     nkmesh2= Nk1*Nk2
     allocate(Omega_x(Num_wann), Omega_y(Num_wann), Omega_z(Num_wann))
     allocate(thetas(nkmesh2), phis(nkmesh2), kpoints(3, nkmesh2))

     !> 在笛卡尔坐标系中设置球面
     ik= 0
     do ik2=1, Nk2 ! θ角循环（相对于z轴的角度）
        theta= (ik2- 1d0)/(Nk2- 1d0)* pi  ! 均匀分布的θ角
        ! 避免在极点处出现数值问题
        if (ik2== 1) theta= (ik2- 1d0+ 0.10)/(Nk2- 1d0)* pi  ! 避免北极
        if (ik2== Nk2) theta= (ik2- 1d0- 0.10)/(Nk2- 1d0)* pi  ! 避免南极
        do ik1=1, Nk1  ! φ角循环（在xy平面内的极角）
           ik = ik+ 1
           phi= (ik1- 1d0)/Nk1* 2d0* pi  ! 均匀分布的φ角
           r_para= r0* sin(theta)  ! 平行圆半径
           ! 计算笛卡尔坐标
           k_cart(1)= k0(1)+ r_para* cos(phi)
           k_cart(2)= k0(2)+ r_para* sin(phi)
           k_cart(3)= k0(3)+ r0* cos(theta)
           ! 转换为分数坐标
           call cart_direct_rec(k_cart, k_direct)
           kpoints(:, ik)= k_direct
           thetas(ik)= theta
           phis(ik)= phi
        enddo
      enddo
      
     ! 计算积分步长
     dtheta= 1.0d0/(Nk2-1d0)*pi   ! θ方向步长
     dphi  = 1.0d0/Nk1 * 2d0*pi    ! φ方向步长

     ! 并行计算陈数
     Chern_mpi = 0d0
     do ik=1+cpuid, Nk1*Nk2, num_cpu
        k_direct= kpoints(:, ik)  ! 当前k点分数坐标
        theta= thetas(ik)         ! 当前θ角
        phi= phis(ik)             ! 当前φ角
        
        ! 计算三角函数值
        st=sin(theta); ct=cos(theta); sp=sin(phi); cp=cos(phi)
        
        ! 计算当前k点的Berry曲率
        call Berry_curvature_singlek_numoccupied(k_direct, Omega_x, Omega_y, Omega_z)
        
        ! 对占据能带求和（这里似乎有重复计算，可能是调试代码）
        O_x= real(sum(Omega_x(1:Numoccupied)))
        O_y= real(sum(Omega_y(1:Numoccupied)))
        O_z= real(sum(Omega_z(1:Numoccupied)))
        O_x= real((Omega_x(Numoccupied)))  ! 只取最后一个占据能带
        O_y= real((Omega_y(Numoccupied)))
        O_z= real((Omega_z(Numoccupied)))
        
        ! 球面积分：∫ sin²θ cosφ dθdφ 等
        Chern_mpi= Chern_mpi+ st*st*cp*O_x+ st*st*sp*O_y+ st*ct*O_z
     enddo

#if defined (MPI)
     ! MPI归约所有进程的陈数贡献
     call mpi_allreduce(Chern_mpi, Chern, 1, &
                       mpi_dp,mpi_sum,mpi_cmw,ierr)
#else
     Chern= Chern_mpi 
#endif

     ! 乘以积分测度得到最终陈数
     Chern = Chern* r0* r0* dtheta* dphi/pi/2d0

     return
  end subroutine Chern_sphere_single

  ! 子程序：计算多个Weyl点的陈数
  subroutine Chern_sphere
     use wmpi
     use para
     implicit none

     integer :: i
     real(dp) :: k0(3), Chern
     real(dp), allocatable :: Chern_array(:)  ! 存储各Weyl点的陈数

     allocate(Chern_array(Num_Weyls))
     Chern_array= 0d0

     ! 遍历所有Weyl点
     do i=1, Num_Weyls
        k0= weyl_position_cart(:, i)  ! 第i个Weyl点的位置
        call Chern_sphere_single(k0, kr0, Chern)  ! 计算该点的陈数
        Chern_array(i)= Chern  ! 存储结果
     enddo

     ! 输出结果
     if (cpuid==0)then
        write(stdout, *)'# Chern number for the Weyl points'
        write(stdout, '("#",a8,2a9, a16)')'kx', 'ky', 'kz', 'Chern'
        do i=1, Num_Weyls
           write(stdout, '(3f9.5, f16.8)')weyl_position_cart(:, i), Chern_array(i)
        enddo
     endif

     return
  end subroutine Chern_sphere

  ! 子程序：在半环面上计算陈数（类似前面的结构，此处省略详细注释）
  subroutine Chern_halftorus_single(k0, Rbig, rsmall_a, rsmall_b, Chern)
     !> 通过在半环面上积分计算陈数
     ! [详细注释类似Chern_sphere_single，主要区别在于几何形状和积分测度]
     use wmpi
     use para
     implicit none
     
     ! [参数和变量声明...]
     ! [实现细节...]
     ! [MPI归约和输出...]

     return
  end subroutine Chern_halftorus_single

  ! 子程序：计算多个非线性点的陈数
  subroutine Chern_halftorus
     use wmpi
     use para
     implicit none

     ! [变量声明和实现...]
     ! 类似于Chern_sphere，但针对非线性点(NL)

     return
  end subroutine Chern_halftorus
```
```fortran

! 子程序：计算给定路径的Berry相位
  subroutine  berryphase
      !> 计算给定路径Berry相位的子程序
      !
      ! 注释：
      !
      !          目前，您必须在kpoints中定义想要的k路径
      !
      ! 作者：QuanSheng Wu (wuquansheng@gmail.com)
      !
      ! 2016年3月31日
      !
      ! 版权所有 (c) 2010 QuanSheng Wu

      use para      ! 引入参数模块
      use wmpi      ! 引入MPI并行模块
      implicit none ! 显式声明所有变量类型

      ! 局部变量声明
      integer :: i, j, it, ik, Nk_seg, NK_Berry_tot  ! 循环计数器和数组维度

      !> k点在kx-ky平面中的位置 (3维坐标, 总k点数)
      real(dp), allocatable :: kpoints(:, :)

      !> 每个k点的哈密顿量
      !> 以及eigensystem_c对角化后的哈密顿量本征向量
      complex(dp), allocatable :: uk(:, :), uk_dag(:, :)  ! uk: 本征向量矩阵, uk_dag: 其厄米共轭

      !> 每个k点的本征向量 (能带×能带×k点)
      complex(dp), allocatable :: Eigenvector(:, :, :)

      ! 本征值数组和相位数组
      real   (dp), allocatable :: eigenvalue(:)     ! 哈密顿量本征值
      complex(dp), allocatable :: phase(:)          ! Berry相位累积因子
      real(dp) :: br                                ! Wannier中心投影的临时变量
      real(dp) :: k(3), b(3)                        ! k点坐标和路径段向量
      complex(dp) :: overlap, ratio                  ! 重叠积分和相位修正因子

      ! 每段k点的数量和Berry相位计算的总k点数
      Nk_seg= Nk                                    ! 每个路径段的k点数
      NK_Berry_tot= (NK_Berry-1)*Nk_seg             ! 闭合路径总k点数

      ! 分配内存
      allocate(kpoints(3, NK_Berry_tot))            ! k点坐标数组
      kpoints= 0d0                                  ! 初始化为零

      ! 分配哈密顿量和本征向量相关数组
      allocate(uk(Num_wann, Num_wann),  uk_dag(Num_wann, Num_wann))  ! 单点哈密顿量和其厄米共轭
      allocate(Eigenvector(Num_wann, Num_wann, NK_Berry_tot))       ! 所有k点的本征向量
      allocate(eigenvalue(Num_wann))                ! 本征值数组
      allocate(phase(Num_wann))                     ! Berry相位累积因子
      
      ! 初始化数组
      uk=0d0
      eigenvalue=0d0
      Eigenvector=0d0

      !> 设置Berry相位计算的k路径
      !> kpoints, k3points_Berry是分数坐标/直接坐标
      it = 0                                        ! k点索引初始化
      do ik=1, NK_Berry- 1                          ! 遍历路径段
         do i= 1, Nk_seg                           ! 在每个路径段内均匀采样
            it= it+ 1
            ! 线性插值构建路径上的k点
            kpoints(:, it)= k3points_Berry(:, ik)+ &
               (k3points_Berry(:, ik+1)- k3points_Berry(:, ik))*(i-1d0)/(Nk_seg-1d0)
         enddo ! i
      enddo ! ik

      !> 对每个k点计算Wannier中心相关的Berry相位
      do ik=1, NK_Berry_tot
         k= kpoints(:, ik)                         ! 当前k点坐标

         ! 根据计算模型选择不同的哈密顿量构造方法
         if (index(KPorTB, 'KP')/=0)then            ! 如果使用kp模型
            call ham_bulk_kp (k, uk)               ! 调用kp模型的体态哈密顿量
         else                                      ! 否则使用晶格规范
            call ham_bulk_latticegauge(k, uk)       ! 调用晶格规范的体态哈密顿量
         endif
        
         !> 对角化哈密顿量获取本征向量
         call eigensystem_c('V', 'U', Num_wann, uk, eigenvalue)

         ! 存储当前k点的本征向量
         Eigenvector(:, :, ik)= uk
      enddo

      !> 沿k路径求和得到Berry相位
      phase= 1d0                                     ! 初始化相位累积因子为1
      do ik= 1, NK_Berry_tot-1                       ! 遍历相邻k点对
         uk= Eigenvector(:, :, ik)                   ! 当前k点的本征向量
         uk_dag= conjg(transpose(uk))                ! 计算其厄米共轭 (bra向量)
         
         ! 处理闭合路径：最后一个点与起点连接
         if (ik==NK_Berry_tot-1) then
            uk= Eigenvector(:, :, 1)                ! 最后一个段连接到起点
         else
            uk= Eigenvector(:, :, ik+ 1)            ! 普通情况连接到下一个点
         endif
         
         b= kpoints(:, ik+1)- kpoints(:, ik)          ! 计算k路径段向量

         !> 计算 <u_k|u_{k+1}> 重叠积分，考虑Wannier中心的规范变换
         do i=1, Num_wann                            ! 遍历能带
            overlap= 0d0                             ! 初始化重叠积分
            
            do j=1, Num_wann                        ! 遍历能带求和
               ! 计算Wannier中心在路径段方向上的投影
               br= b(1)*Origin_cell%wannier_centers_direct(1, j)+ &
                   b(2)*Origin_cell%wannier_centers_direct(2, j)+ &
                   b(3)*Origin_cell%wannier_centers_direct(3, j)
               ! 规范变换相位因子：exp(i2π br)
               ratio= cos(2d0*pi*br)- zi*sin(2d0*pi*br)

               ! 计算矩阵元素乘积并累加
               overlap= overlap+ uk_dag(i, j)* uk(j, i)* ratio
            enddo
            
            ! 累积相位因子
            phase(i)= overlap*phase(i)
         enddo

      enddo  !< ik 完成所有路径段的积分

      ! 输出警告信息和计算结果
      if (cpuid==0)write(stdout, *) ">> WARNING: Please increase NK1 until Berry phase is converged!"
      if (cpuid==0)write(stdout, *) ">> WARNING: The starting point and the ending point should be different by a reciprocal lattice vector"
      if (cpuid==0)write(stdout, *) 'Berry phase for the loop you chose: in unit of \pi'
      ! 计算占据能带的Berry相位（模2π，以π为单位）
      if (cpuid==0) write(stdout, '(f18.6)') mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0)

      ! 输出详细数据到文件
      outfileindex= outfileindex+ 1
      if (cpuid==0) then
         open(unit= outfileindex, file="kpath_berry.txt")
         ! 写入文件头，包含Berry相位结果
         write(outfileindex, '("#",a11, 5a12, a, f12.6, a)')"kx", "ky", "kz", &
                              "k1", "k2", "k3", " Berry phase= ", &
                              mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0), ' pi'
         ! 写入所有k点的坐标
         do ik=1, NK_Berry_tot
            ! 将分数坐标转换为笛卡尔坐标
            b= kpoints(1, ik)*Origin_cell%Kua+ kpoints(2, ik)*Origin_cell%Kub+ kpoints(3, ik)*Origin_cell%Kuc
            write(outfileindex, '(6f12.6)')b, kpoints(:, ik)
         enddo
      endif
      
      ! 释放动态分配的内存
      deallocate(kpoints)
      deallocate(uk,uk_dag)
      deallocate(Eigenvector,eigenvalue)
      deallocate(phase)
 
      return
   end subroutine berryphase

  ! 子程序：使用原子规范计算Berry相位（改进版本）
  subroutine  berryphase_atomic
      !> 使用原子规范计算给定路径Berry相位的子程序
      ! [头部注释与上一子程序相同，此处省略]

      use para
      use wmpi
      implicit none

      ! 局部变量（与上一子程序类似，增加mat1, mat2）
      integer :: i, j, it, ik, Nk_seg, NK_Berry_tot

      !> k点在kx-ky平面中的位置
      real(dp), allocatable :: kpoints(:, :)

      !> 哈密顿量和本征向量
      complex(dp), allocatable :: uk(:, :), uk_dag(:, :)
      complex(dp), allocatable :: Eigenvector(:, :, :)

      ! 本征值和相关矩阵
      real   (dp), allocatable :: eigenvalue(:)
      complex(dp), allocatable :: phase(:), mat1(:, :), mat2(:, :)  ! 新增工作矩阵
      real(dp) :: br
      real(dp) :: k(3), b(3)
      complex(dp) :: overlap, ratio

      ! 路径参数设置（与上一子程序相同）
      Nk_seg= Nk
      NK_Berry_tot= (NK_Berry-1)*Nk_seg 

      ! 分配内存（额外分配mat1, mat2用于中间计算）
      allocate(kpoints(3, NK_Berry_tot))
      kpoints= 0d0

      allocate(mat1(Num_wann, Num_wann), mat2(Num_wann, Num_wann))  ! 工作矩阵
      allocate(uk(Num_wann, Num_wann),  uk_dag(Num_wann, Num_wann))
      allocate(Eigenvector(Num_wann, Num_wann, NK_Berry_tot), eigenvalue(Num_wann))
      allocate(phase(Num_wann))
      
      ! 初始化
      uk=0d0
      eigenvalue=0d0
      Eigenvector=0d0

      !> 设置k路径（与上一子程序完全相同）
      it = 0
      do ik=1, NK_Berry- 1
         do i= 1, Nk_seg
            it= it+ 1
            kpoints(:, it)= k3points_Berry(:, ik)+ &
               (k3points_Berry(:, ik+1)- k3points_Berry(:, ik))*(i-1d0)/(Nk_seg-1d0)
         enddo ! i
      enddo ! ik

      !> 计算每个k点的本征向量（关键区别：使用原子规范）
      do ik=1, NK_Berry_tot
         k= kpoints(:, ik)

         ! 主要区别：使用原子规范而非晶格规范或kp模型
         if (index(KPorTB, 'KP')/=0)then
            call ham_bulk_kp (k, uk)               ! kp模型（如果适用）
         else
            call ham_bulk_atomicgauge(k, uk)       ! 原子规范哈密顿量
         endif
        
         !> 对角化（与上一子程序相同）
         call eigensystem_c('V', 'U', Num_wann, uk, eigenvalue)

         Eigenvector(:, :, ik)= uk
      enddo

      !> 计算Berry相位（主要逻辑改进）
      phase= 1d0
      do ik= 1, NK_Berry_tot-1
         uk= Eigenvector(:, :, ik)
         uk_dag= conjg(transpose(uk))
         
         ! 改进的路径闭合处理
         if (ik==NK_Berry_tot-1) then
            ! 显式计算闭合向量（从终点到起点）
            b=-kpoints(:, 1)+ kpoints(:, NK_Berry_tot)
            uk= Eigenvector(:, :, 1)
         else
            uk= Eigenvector(:, :, ik+ 1)
         endif

         !> 改进的重叠积分计算
         do i=1, Num_wann
            overlap= 0d0
            do j=1, Num_wann
               if (ik==NK_Berry_tot-1) then
                  ! 只在闭合段应用Wannier中心规范变换
                  br= b(1)*Origin_cell%wannier_centers_direct(1, j)+ &
                      b(2)*Origin_cell%wannier_centers_direct(2, j)+ &
                      b(3)*Origin_cell%wannier_centers_direct(3, j)
                  ratio= cos(2d0*pi*br)- zi*sin(2d0*pi*br)
               else
                  ! 非闭合段不使用规范变换（ratio=1）
                  ratio=1d0
               endif

               overlap= overlap+ uk_dag(i, j)* uk(j, i)* ratio
            enddo
            phase(i)= overlap*phase(i)
         enddo

      enddo  !< ik

      ! 输出部分（与上一子程序完全相同）
      if (cpuid==0)write(stdout, *) ">> WARNING: Please increase NK1 until Berry phase is converged!"
      if (cpuid==0)write(stdout, *) ">> WARNING: The starting point and the ending point should be different by a reciprocal lattice vector"
      if (cpuid==0)write(stdout, *) 'Berry phase for the loop you chose: in unit of \pi'
      if (cpuid==0) write(stdout, '(f18.6)') mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0)

      outfileindex= outfileindex+ 1
      if (cpuid==0) then
         open(unit= outfileindex, file="kpath_berry.txt")
         write(outfileindex, '("#",a11, 5a12, a, f12.6, a)')"kx", "ky", "kz", &
                              "k1", "k2", "k3", " Berry phase= ", &
                              mod(sum(aimag(log(phase(1:NumOccupied)))/pi), 2d0), ' pi'
         do ik=1, NK_Berry_tot
            b= kpoints(1, ik)*Origin_cell%Kua+ kpoints(2, ik)*Origin_cell%Kub+ kpoints(3, ik)*Origin_cell%Kuc
            write(outfileindex, '(6f12.6)')b, kpoints(:, ik)
         enddo
      endif
      
      ! 释放内存（包括新增的mat1, mat2）
      deallocate(kpoints)
      deallocate(mat1,mat2,uk,uk_dag)
      deallocate(Eigenvector,eigenvalue)
      deallocate(phase)
 
      return
   end subroutine berryphase_atomic

```