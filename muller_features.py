# 定义偏振参数计算
from scipy.ndimage import uniform_filter
from scipy.linalg import logm
import numpy as np

parameters=['FinalM11', 'FinalM12', 'FinalM13', 'FinalM14',
            'FinalM21', 'FinalM22', 'FinalM23', 'FinalM24',
            'FinalM31', 'FinalM32', 'FinalM33', 'FinalM34',
            'FinalM41', 'FinalM42', 'FinalM43', 'FinalM44',]

# Translate matrix elements to index number
def M11(MM): return MM[..., 0, 0]
def M12(MM): return MM[..., 0, 1]
def M13(MM): return MM[..., 0, 2]
def M14(MM): return MM[..., 0, 3]
def M21(MM): return MM[..., 1, 0]
def M22(MM): return MM[..., 1, 1]
def M23(MM): return MM[..., 1, 2]
def M24(MM): return MM[..., 1, 3]
def M31(MM): return MM[..., 2, 0]
def M32(MM): return MM[..., 2, 1]
def M33(MM): return MM[..., 2, 2]
def M34(MM): return MM[..., 2, 3]
def M41(MM): return MM[..., 3, 0]
def M42(MM): return MM[..., 3, 1]
def M43(MM): return MM[..., 3, 2]
def M44(MM): return MM[..., 3, 3]

### remember always call clear_all_MMD() before load a new data
def clear_all_MMD():
    global MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_,MMD_not_calculated
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    global MMLD_Lm
    global MMLD_Lu


    MMCD_not_calculated=True
    MMCD_lambdas = np.array([])
    MMCD_Ps = np.array([])
    MMLD_Lm = np.array([])
    MMLD_Lu = np.array([])
    
    MMD_not_calculated=True

    MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=0.,0.,0.,0.,0.,0.

############################################################################################
# Mueller Matrix Transformation (MMT) parameters
############################################################################################



### B is the central 2x2 block of Mueller matrix

def B_b(MM):
    return (M22(MM)+M33(MM))/2.

def B_beta(MM):
    return (M23(MM)-M32(MM))/2.

def B_b_tld(MM):
    return (M22(MM)-M33(MM))/2.

def B_beta_tld(MM):
    return (M23(MM)+M32(MM))/2.

def t1(MM):
    b_tld_ = B_b_tld(MM)
    beta_tld_ = B_beta_tld(MM)
    return np.sqrt(b_tld_**2 + beta_tld_**2)/2.

def MMT_A(MM):
    b_= B_b(MM)
    t1_= t1(MM)
    return 2*b_*t1_/(b_**2+t1_**2)

def B_det(MM):
    return M22(MM)*M33(MM)-M23(MM)*M32(MM)

def B_norm(MM):
    return np.sqrt(M22(MM)**2+M33(MM)**2+M23(MM)**2+M32(MM)**2)

def alpha_1(MM):
    return np.arctan2(M23(MM)+M32(MM), M22(MM)-M33(MM))/4


### corner parameters of Mueller matrix

def CD(MM):
    return (M14(MM)+M41(MM))

### edge parameters of Mueller matrix

def P_L(MM):
    return np.sqrt(M21(MM)**2+M31(MM)**2)

def D_L(MM):
    return np.sqrt(M12(MM)**2+M13(MM)**2)

def r_L(MM):
    return np.sqrt(M24(MM)**2+M34(MM)**2)

def q_L(MM):
    return np.sqrt(M42(MM)**2+M43(MM)**2)

### edge orientation parameters of Mueller matrix

def alpha_P(MM):
    return np.arctan2(M31(MM),M21(MM))/2

def alpha_D(MM):
    return np.arctan2(M13(MM),M12(MM))/2

def alpha_r(MM):
    return np.arctan2(-M24(MM),M34(MM))/2

def alpha_q(MM):
    return np.arctan2(M42(MM),-M43(MM))/2


### orientation difference parameters of Mueller matrix

# @jit(nopython=True)
def alpha_range_limit(A):
    B = A
    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i,j] > np.pi/2:
                B[i,j] -= np.pi
            elif B[i,j] < -np.pi/2:
                B[i,j] += np.pi
    return B


def alpha_DP_dif(MM):
    return alpha_range_limit(alpha_D(MM) - alpha_P(MM))


def alpha_rq_dif(MM):
    return alpha_range_limit(alpha_r(MM) - alpha_q(MM))


def alpha_DP_cos(MM):
    return (M12(MM)*M21(MM)+M13(MM)*M31(MM))/(P_L(MM)*D_L(MM))

def alpha_rq_cos(MM):
    return (M24(MM)*M42(MM)+M34(MM)*M43(MM))/(r_L(MM)*q_L(MM))

def alpha_DP_sin(MM):
    return (M12(MM)*M31(MM)-M13(MM)*M21(MM))/(P_L(MM)*D_L(MM))

def alpha_rq_sin(MM):
    return (M24(MM)*M43(MM)-M34(MM)*M42(MM))/(r_L(MM)*q_L(MM))

def transpose_asym_DP(MM):
    return np.sqrt((M12(MM)-M21(MM))**2+(M13(MM)-M31(MM))**2)

def transpose_asym_rq(MM):
    return np.sqrt((M24(MM)+M42(MM))**2+(M34(MM)+M43(MM))**2)

############################################################################################
# Gil invariant parmeters
############################################################################################


def MM_Det(A):
    return np.linalg.det(A)


def MM_Norm(A):
    return np.linalg.norm(A,axis =(2,3))


def MM_Trace(MM):
    return (M11(MM)+M22(MM)+M33(MM)+M44(MM))


def P_vec(MM):
    return np.sqrt(M21(MM)**2+M31(MM)**2+M41(MM)**2)

def D_vec(MM):
    return np.sqrt(M12(MM)**2+M13(MM)**2+M14(MM)**2)

def P_dot_D(MM):
    return (M12(MM)*M21(MM)+M13(MM)*M31(MM)+M14(MM)*M41(MM))

def P_m_D(A):
    B = np.zeros(A.shape[:2])
    for i in range(len(A)):
        for j in range(len(A[0])):
            B[i,j] = np.matmul(np.transpose(A[i,j,1:,0:1]), np.matmul(A[i,j,1:,1:], np.transpose(A[i,j,0:1,1:])))[0,0]
    return B

def P_mT_D(A):
    B = np.zeros(A.shape[:2])
    for i in range(len(A)):
        for j in range(len(A[0])):
            B[i,j] = np.matmul(np.matmul(A[i,j,0:1,1:],A[i,j,1:,1:]), A[i,j,1:,0:1])[0,0]
    return B

############################################################################################
# Cloude decomposition parameters
############################################################################################
MMCD_not_calculated=True

def check_MMCD(MM):
    global MMCD_lambdas, MMCD_Ps, MMCD_not_calculated

    print("calc MMCD triggered")
    H = H_cloude(MM)

    MMCD_lambdas = calc_MMCD(H)

    MMCD_Ps = calc_MMCD_Ps(H, MMCD_lambdas)
    MMCD_not_calculated=False
    return MMCD_lambdas,MMCD_Ps


def H_cloude(MM):
    H = np.zeros(MM.shape, dtype=np.complex64)

    H[:,:,0,0] = MM[:,:,0,0] + MM[:,:,0,1] +     MM[:,:,1,0] + MM[:,:,1,1]
    H[:,:,0,1] = MM[:,:,0,2] + MM[:,:,1,2] + 1j*(MM[:,:,0,3] + MM[:,:,1,3])
    H[:,:,0,2] = MM[:,:,2,0] + MM[:,:,2,1] - 1j*(MM[:,:,3,0] + MM[:,:,3,1])
    H[:,:,0,3] = MM[:,:,2,2] + MM[:,:,3,3] + 1j*(MM[:,:,2,3] - MM[:,:,3,2])
    H[:,:,1,0] = MM[:,:,0,2] + MM[:,:,1,2] - 1j*(MM[:,:,0,3] + MM[:,:,1,3])
    H[:,:,1,1] = MM[:,:,0,0] - MM[:,:,0,1] +     MM[:,:,1,0] - MM[:,:,1,1]
    H[:,:,1,2] = MM[:,:,2,2] - MM[:,:,3,3] - 1j*(MM[:,:,2,3] + MM[:,:,3,2])
    H[:,:,1,3] = MM[:,:,2,0] - MM[:,:,2,1] - 1j*(MM[:,:,3,0] - MM[:,:,3,1])
    H[:,:,2,0] = MM[:,:,2,0] + MM[:,:,2,1] + 1j*(MM[:,:,3,0] + MM[:,:,3,1])
    H[:,:,2,1] = MM[:,:,2,2] - MM[:,:,3,3] + 1j*(MM[:,:,2,3] + MM[:,:,3,2])
    H[:,:,2,2] = MM[:,:,0,0] + MM[:,:,0,1] -     MM[:,:,1,0] - MM[:,:,1,1]
    H[:,:,2,3] = MM[:,:,0,2] - MM[:,:,1,2] + 1j*(MM[:,:,0,3] - MM[:,:,1,3])
    H[:,:,3,0] = MM[:,:,2,2] + MM[:,:,3,3] - 1j*(MM[:,:,2,3] - MM[:,:,3,2])
    H[:,:,3,1] = MM[:,:,2,0] - MM[:,:,2,1] + 1j*(MM[:,:,3,0] - MM[:,:,3,1])
    H[:,:,3,2] = MM[:,:,0,2] - MM[:,:,1,2] - 1j*(MM[:,:,0,3] - MM[:,:,1,3])
    H[:,:,3,3] = MM[:,:,0,0] - MM[:,:,0,1] -     MM[:,:,1,0] + MM[:,:,1,1] 
    return H/4

def calc_MMCD(H):
    return np.real(np.sort(np.linalg.eigvals(H))[:,:,::-1])

def calc_MMCD_Ps(H, Ls):
    B = np.zeros((Ls.shape[0],Ls.shape[1],11),dtype=np.complex64)

    T = H[:,:, 0, 0] + H[:,:, 1, 1] + H[:,:, 2, 2] + H[:,:, 3, 3]
    L = Ls/np.tensordot(T,np.array([1,1,1,1]),axes=0)
    P1 = (L[:,:,0] - L[:,:,1])
    P2 = (L[:,:,0] + L[:,:,1] - 2*L[:,:,2])
    P3 = (L[:,:,0] + L[:,:,1] + L[:,:,2] - 3*L[:,:,3])
    PI = np.sqrt((P1**2 + P2**2 + P3**2)/3)
    PD = np.sqrt((2 * P1**2 + 2 / 3 * P2**2 + 1 / 3 * P3**2)/3)
    S = -(L[:,:,0]*np.log2(L[:,:,0]) + L[:,:,1]*np.log2(L[:,:,1]) + L[:,:,2]*np.log2(L[:,:,2]) + L[:,:,3]*np.log2(L[:,:,3]))/2
    B[:,:,:4]=L
    B[:,:,4:]=np.stack([T,P1,P2,P3,PI,PD,S],axis=2)
    return B


def MMCD_Lambda1(MM):
    global MMCD_lambdas, MMCD_Ps, MMCD_not_calculated

    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,0])

def MMCD_Lambda2(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,1])

def MMCD_Lambda3(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,2])

def MMCD_Lambda4(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,3])

def MMCD_P1(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,5])

def MMCD_P2(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,6])

def MMCD_P3(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,7])


def MMCD_PI(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,8])

def MMCD_PD(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated

    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,9])

def MMCD_S(MM):
    global MMCD_lambdas, MMCD_Ps,MMCD_not_calculated
    if MMCD_not_calculated:
        MMCD_lambdas,MMCD_Ps=check_MMCD(MM)
    return (MMCD_Ps[:,:,10])

############################################################################################
# Differential decomposition parameters
############################################################################################

def check_MMLD(MM):
    global MMLD_Lm
    global MMLD_Lu

    # remember always call clear_all_MMD() before load a new data
    if(MMLD_Lm.shape[:2] != MM.shape[:2]):
        print("calc MMLD triggered")
        [MMLD_Lm, MMLD_Lu] = calc_MMLD(MM)#calc_MMLD(MM)

def calc_MMLD(MM):
    temp=uniform_filter(MM[:,:,0,0],10)[::10,::10]
    A=np.zeros((temp.shape[0],temp.shape[1],4,4))
    for i in range(4):
        for j in range(4):
            A[:,:,i,j]=uniform_filter(MM[:,:,i,j],10)[::10,::10]
        
    L = np.zeros( (A.shape[0],A.shape[1],4,4) )
    G =np.zeros( (A.shape[0],A.shape[1],4,4) )+ np.diag([1,-1,-1,-1])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            L[i,j]=logm(A[i,j])
    T=np.matmul(G, np.matmul((L).transpose((0,1,3,2)), G))
    Lm = (L-T)/2
    Lu = (L+T)/2
    Lu=Lu-Lu[:,:,[[0]],[[0]]]*(np.zeros( (A.shape[0],A.shape[1],4,4) )+ np.eye(4))        

    Lu_fin=np.zeros(MM.shape)
    Lm_fin=np.zeros(MM.shape)
    for i in range(4):
        for j in range(4):
            Lu_resized = np.resize(Lu[:,:,i,j], (MM.shape[0], MM.shape[1]))
            Lm_resized = np.resize(Lm[:,:,i,j], (MM.shape[0], MM.shape[1]))
            Lu_fin[:,:,i,j]=Lu_resized
            Lm_fin[:,:,i,j]=Lm_resized

    return np.array([Lm_fin,Lu_fin])



def MMLD_D(MM):
    check_MMLD(MM)
    return np.sqrt(MMLD_Lm[:,:,0,1]**2+MMLD_Lm[:,:,0,2]**2)


def MMLD_delta(MM):
    check_MMLD(MM)
    return np.sqrt(MMLD_Lm[:,:,1,3]**2+MMLD_Lm[:,:,2,3]**2)


def MMLD_alpha(MM):
    check_MMLD(MM)
    return (MMLD_Lm[:,:,1,2]/2)


def MMLD_CD(MM):
    check_MMLD(MM)
    return (MMLD_Lm[:,:,0,3])


def MMLD_a22(MM):
    check_MMLD(MM)
    return (MMLD_Lu[:,:,1,1])


def MMLD_a33(MM):
    check_MMLD(MM)
    return (MMLD_Lu[:,:,2,2])


def MMLD_aL(MM):
    check_MMLD(MM)
    return (MMLD_Lu[:,:,1,1]+MMLD_Lu[:,:,2,2])/2


def MMLD_a44(MM):
    check_MMLD(MM)
    return (MMLD_Lu[:,:,3,3])


def MMLD_aLA(MM):
    check_MMLD(MM)
    return np.sqrt((MMLD_Lu[:,:,1,1]-MMLD_Lu[:,:,2,2])**2 + (MMLD_Lu[:,:,1,2]+MMLD_Lu[:,:,2,1])**2)/2.

############################################################################################
# Equalities parameters
############################################################################################

def Es(MM):
    return (M11(MM)-M22(MM)-M33(MM)+M44(MM))

def E1(MM):
    return ((M11(MM)+M22(MM))**2-(M12(MM)+M21(MM))**2-(M33(MM)+M44(MM))**2-(M34(MM)-M43(MM))**2)

def E2(MM):
    return ((M11(MM)-M22(MM))**2-(M12(MM)-M21(MM))**2-(M33(MM)-M44(MM))**2-(M34(MM)+M43(MM))**2)

def E3(MM):
    return ((M11(MM)+M21(MM))**2-(M12(MM)+M22(MM))**2-(M13(MM)+M23(MM))**2-(M14(MM)+M24(MM))**2)

def E4(MM):
    return ((M11(MM)-M21(MM))**2-(M12(MM)-M22(MM))**2-(M13(MM)-M23(MM))**2-(M14(MM)-M24(MM))**2)

def E5(MM):
    return ((M11(MM)+M12(MM))**2-(M21(MM)+M22(MM))**2-(M31(MM)+M32(MM))**2-(M41(MM)+M42(MM))**2)

def E6(MM):
    return ((M11(MM)-M12(MM))**2-(M21(MM)-M22(MM))**2-(M31(MM)-M32(MM))**2-(M41(MM)-M42(MM))**2)

def cal_MMD(MM):
    D=np.sqrt(MM[:,:,0,1]**2+MM[:,:,0,2]**2+MM[:,:,0,3]**2)
    D_vector=np.zeros((MM.shape[0],MM.shape[1],3,1))
    for i in range(3):
        D_vector[:,:,i,0]=MM[:,:,0,i+1]    
    D_vector_hat=D_vector/(D.reshape(D.shape[0],D.shape[1],1,1))
    m_D=np.sqrt(1-D**2).reshape(MM.shape[0],MM.shape[1],1,1)*(np.zeros((MM.shape[0],MM.shape[1],3,3))+np.eye(3))+(1-np.sqrt(1-D**2).reshape(MM.shape[0],MM.shape[1],1,1))*np.matmul(D_vector_hat,D_vector_hat[:,:,:,0].reshape(MM.shape[0],MM.shape[1],1,3))
    M_D=np.zeros((MM.shape[0],MM.shape[1],4,4))
    M_D[:,:,0,0]=1.
    M_D[:,:,0,1:]=D_vector[:,:,:,0]
    M_D[:,:,1:,0]=D_vector[:,:,:,0]
    M_D[:,:,1:,1:]=m_D

    M_plus=np.matmul(MM,np.linalg.inv(M_D))
    m_plus=M_plus[:,:,1:,1:]
    temp_m_plus=np.matmul(m_plus,m_plus.transpose((0,1,3,2)))
    lamda=np.linalg.eigvals(temp_m_plus)
    m_delta=np.matmul(np.linalg.inv(temp_m_plus+(np.sqrt(lamda[:,:,[[0]]]*lamda[:,:,[[1]]])+np.sqrt(lamda[:,:,[[1]]]*lamda[:,:,[[2]]])+
                                    np.sqrt(lamda[:,:,[[0]]]*lamda[:,:,[[2]]]))*(np.zeros((MM.shape[0],MM.shape[1],3,3))+np.eye(3))),
                               (np.sqrt(lamda[:,:,[[0]]])+np.sqrt(lamda[:,:,[[1]]])+np.sqrt(lamda[:,:,[[2]]]))*temp_m_plus 
                   +np.sqrt(lamda[:,:,[[0]]]*lamda[:,:,[[1]]]*lamda[:,:,[[2]]])*(np.zeros((MM.shape[0],MM.shape[1],3,3))+np.eye(3))
                   *np.sign(np.linalg.det(m_plus)).reshape(MM.shape[0],MM.shape[1],1,1))
    delta=1-abs(m_delta[:,:,0,0]+m_delta[:,:,1,1]+m_delta[:,:,2,2])/3
    m_R=np.matmul(np.linalg.inv(m_delta),m_plus)
    R=np.arccos((1+m_R[:,:,0,0]+m_R[:,:,1,1]+m_R[:,:,2,2])/2-1)
    ita=np.arccos(np.sqrt((m_R[:,:,0,0]+m_R[:,:,1,1])**2+(m_R[:,:,1,0]-m_R[:,:,0,1])**2)-1)
    psi=0.5*np.arctan2((m_R[:,:,1,0]-m_R[:,:,0,1]),(m_R[:,:,0,0]+m_R[:,:,1,1]))

    m_psi=np.zeros((MM.shape[0],MM.shape[1],3,3))
    m_psi[:,:,0,0]=np.cos(2*psi)
    m_psi[:,:,0,1]=np.sin(2*psi)
    m_psi[:,:,0,2]=0
    m_psi[:,:,1,0]=-np.sin(2*psi)
    m_psi[:,:,1,1]=np.cos(2*psi)
    m_psi[:,:,1,2]=0
    m_psi[:,:,2,1]=0
    m_psi[:,:,2,1]=0
    m_psi[:,:,2,2]=1

    m_LR=np.matmul(m_R,np.linalg.inv(m_psi))
    r1=1./(2*np.sin(ita))*(m_LR[:,:,1,2]-m_LR[:,:,2,1])
    r2=1./(2*np.sin(ita))*(m_LR[:,:,2,0]-m_LR[:,:,0,2])
    r1=np.real(r1)
    r2=np.real(r2)
    theta=(0.5* np.arctan2(r2,r1) *180/np.pi)
    return np.array([abs(D),abs(delta),abs(ita),abs(R),abs(psi),theta])

MMD_not_calculated=True

MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=0.,0.,0.,0.,0.,0.


def MMD_D(MM):
    global MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_,MMD_not_calculated

    if MMD_not_calculated:
        print('calcuating MMD')
        MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=cal_MMD(MM)
        MMD_not_calculated=False
    return MMD_D_
def MMD_delta(MM):
    global MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_,MMD_not_calculated
    if MMD_not_calculated:
        print('calcuating MMD')
        MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=cal_MMD(MM)
        MMD_not_calculated=False
    return MMD_delta_
def MMD_ita(MM):
    global MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_,MMD_not_calculated
    if MMD_not_calculated:
        print('calcuating MMD')
        MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=cal_MMD(MM)
        MMD_not_calculated=False 
    return MMD_ita_
def MMD_R(MM):
    global MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_,MMD_not_calculated
    if MMD_not_calculated:
        print('calcuating MMD')
        MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=cal_MMD(MM)
        MMD_not_calculated=False
    return MMD_R_
def MMD_psi(MM):
    global MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_,MMD_not_calculated
    if MMD_not_calculated:
        print('calcuating MMD')
        MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=cal_MMD(MM)
        MMD_not_calculated=False
    return MMD_psi_
def MMD_theta(MM):
    global MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_,MMD_not_calculated
    if MMD_not_calculated:
        print('calcuating MMD')
        MMD_D_,MMD_delta_,MMD_ita_,MMD_R_,MMD_psi_,MMD_theta_=cal_MMD(MM)
        MMD_not_calculated=False
    return MMD_theta_

function_list_MMT = [B_b,B_beta,B_b_tld,B_beta_tld,t1,MMT_A,B_det,B_norm,alpha_1,CD,P_L,
     D_L,r_L,q_L,alpha_P,alpha_D,alpha_r,alpha_q,alpha_DP_dif,alpha_rq_dif,
     alpha_DP_cos,alpha_rq_cos,alpha_DP_sin,alpha_rq_sin,transpose_asym_DP,
     transpose_asym_rq]
function_list_Gil = [MM_Det, MM_Norm, MM_Trace, P_vec, D_vec, P_dot_D, P_m_D, P_mT_D]
function_list_MMCD = [MMCD_Lambda1, MMCD_Lambda2, MMCD_Lambda3, MMCD_Lambda4, MMCD_P1, MMCD_P2, MMCD_P3, MMCD_PI, MMCD_PD, MMCD_S]
function_list_MMLD = [MMLD_D, MMLD_delta, MMLD_alpha, MMLD_CD, MMLD_a22, MMLD_a33, MMLD_aL, MMLD_a44, MMLD_aLA]
function_list_Eq = [Es, E1, E2, E3, E4, E5, E6]
function_list_MMPD=[MMD_D,MMD_delta,MMD_ita,MMD_R,MMD_psi,MMD_theta]

function_list = function_list_MMT + function_list_Gil + function_list_MMCD + function_list_MMLD + function_list_Eq + function_list_MMPD



import students_submission
if hasattr(students_submission, "function_list") and isinstance(students_submission.function_list, list):
    function_list = students_submission.function_list

function_list_names = [f.__name__ for f in function_list]
if __name__ == "__main__":
    print(function_list_names)
